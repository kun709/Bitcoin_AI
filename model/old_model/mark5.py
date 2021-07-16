import torch
import torch.nn as nn


def activation_function():
    return nn.LeakyReLU()


def downsample(channel, expansion=2, stride=2):
    return nn.Sequential(
        nn.MaxPool1d(2),
        nn.BatchNorm1d(channel),
        # activation_function(),
        nn.Conv1d(channel, channel * expansion, kernel_size=stride, stride=stride)
    )


class Conv1DBlock(nn.Module):
    def __init__(self, channel, kernal=3, padding=1):
        super(Conv1DBlock, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(channel),
            activation_function(),
            nn.Conv1d(channel, channel, kernal, padding=padding, bias=False),
            nn.BatchNorm1d(channel),
            activation_function(),
            nn.Conv1d(channel, channel, kernal, padding=padding, bias=False),
        )

    def forward(self, x):
        return x + self.net(x)


def basic_block(channel, pooling=True):
    if pooling:
        return nn.Sequential(
            Conv1DBlock(channel),
            Conv1DBlock(channel),
            downsample(channel),
            Conv1DBlock(channel * 2),
            Conv1DBlock(channel * 2),
            downsample(channel * 2),
            nn.AdaptiveAvgPool1d(1)
        )
    else:
        return nn.Sequential(
            Conv1DBlock(channel),
            Conv1DBlock(channel),
            downsample(channel),
            Conv1DBlock(channel * 2),
            Conv1DBlock(channel * 2),
            downsample(channel * 2)
        )


class Mark(nn.Module):
    def __init__(self, args):
        super(Mark, self).__init__()
        self.args = args
        self.device = args.device

        self.net_start = nn.Conv1d(5, args.channel, args.kernal_size, bias=False)

        self.net_256 = basic_block(args.channel)
        self.net_128 = basic_block(args.channel)
        self.net_64 = basic_block(args.channel)
        self.net_32 = basic_block(args.channel)
        self.net_16 = basic_block(args.channel, False)

        self.fc = nn.Linear(args.channel * 4 * 3, args.predict_size * 2, bias=False)

    def forward(self, price, volume):
        price = (price + 1) / 2
        x = torch.cat([price, volume.unsqueeze(-1)], dim=2).permute(0, 2, 1)

        x_256 = self.net_start(x)
        x_128 = x_256[:, :, -128:]
        x_64 = x_256[:, :, -64:]
        x_32 = x_256[:, :, -32:]
        x_16 = x_256[:, :, -16:]

        x_256 = self.net_256(x_256)
        x_128 = self.net_128(x_128)
        x_64 = self.net_64(x_64)
        x_32 = self.net_32(x_32)
        x_16 = self.net_16(x_16)

        x = torch.cat([x_256 + x_128, x_64 + x_32, x_16], dim=1).squeeze(2)
        x = self.fc(x).view(-1, 2, self.args.predict_size)
        return x

    def save_param(self, file_name='./param/param.pth'):
        torch.save(self.state_dict(), file_name)

    def load_param(self, file_name='./param/param.pth'):
        self.load_state_dict(torch.load(file_name, map_location=self.args.device))
