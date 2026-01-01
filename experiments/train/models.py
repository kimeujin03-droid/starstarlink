from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    """A small U-Net suitable for 512x512 training on Colab.

    - Image head: 1 channel (pred clean)
    - Optional mask head: K channels (logits)
    """

    def __init__(self, in_ch=1, base=32, mask_channels=0):
        super().__init__()
        self.mask_channels = int(mask_channels)

        self.e1 = ConvBlock(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.e2 = ConvBlock(base, base*2)
        self.p2 = nn.MaxPool2d(2)
        self.e3 = ConvBlock(base*2, base*4)
        self.p3 = nn.MaxPool2d(2)
        self.b  = ConvBlock(base*4, base*8)

        self.u3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.d3 = ConvBlock(base*8, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = ConvBlock(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = ConvBlock(base*2, base)

        self.head_img = nn.Conv2d(base, 1, 1)
        if self.mask_channels > 0:
            self.head_msk = nn.Conv2d(base, self.mask_channels, 1)
        else:
            self.head_msk = None

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        b  = self.b(self.p3(e3))

        d3 = self.d3(torch.cat([self.u3(b), e3], dim=1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], dim=1))

        out_img = self.head_img(d1)
        if self.head_msk is None:
            return out_img
        out_msk = self.head_msk(d1)
        return out_img, out_msk
