import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNNEncoder(nn.Module):

    def __init__(self, input_dim, embed_dim):
        super(CRNNEncoder, self).__init__()

        self.features = nn.Sequential(
            # Conv2D block
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # LPPool
            nn.LPPool2d(4, (2, 4)),

            # Conv2D block
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # Conv2D block
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # LPPool
            nn.LPPool2d(4, (2, 4)),

            # Conv2D block
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # Conv2D block
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),

            # LPPool
            nn.LPPool2d(4, (1, 4)),

            nn.Dropout(0.3)
        )

        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500, input_dim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim, embed_dim // 2, bidirectional=True, batch_first=True)

        self.features.apply(init_weights)

    def forward(self, x, upsample=True):
        """
        :param x: tensor, (batch_size, time_steps, Mel_bands).
        :param upsample: default True.
        :return: (batch_size, time_steps, embed_dim).
        """
        batch, time, dim = x.shape

        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)

        if upsample:
            x = torch.nn.functional.interpolate(x.transpose(1, 2),
                                                time, mode="linear", align_corners=False).transpose(1, 2)

        return x


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
