import torch
from torch import nn
from .modules import WDSRBlock

class WDSRModel(nn.Module):
    def __init__(self, config):
        super(WDSRModel, self).__init__()
        # hyper-params
        self.config = config
        scale = config.scale
        n_resblocks = config.n_resblocks
        n_feats = config.n_feats
        n_colors = 1
        kernel_size = 3
        act = nn.ReLU(True)
        # wn = lambda x: x
        wn = lambda x: torch.nn.utils.weight_norm(x)

#         self.rgb_mean = torch.autograd.Variable(torch.FloatTensor(
#             [config.r_mean, config.g_mean, config.b_mean])).view([1, 3, 1, 1])

        # define head module
        head = []
        head.append(
            wn(nn.Conv2d(n_colors, n_feats, 3, padding=3//2)))

        # define body module
        body = []
        for i in range(n_resblocks):
            body.append(
                WDSRBlock(n_feats, kernel_size, act=act, res_scale=config.res_scale, wn=wn))

        # define tail module
        tail = []
        out_feats = scale*scale*n_colors
        tail.append(
            wn(nn.Conv2d(n_feats, out_feats, 3, padding=3//2)))
        tail.append(nn.PixelShuffle(scale))

        skip = []
        skip.append(
            wn(nn.Conv2d(n_colors, out_feats, 5, padding=5//2))
        )
        skip.append(nn.PixelShuffle(scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
#         x = (x - self.rgb_mean.cuda()*255)/127.5
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
#         x = x*127.5 + self.rgb_mean.cuda()*255
        return x