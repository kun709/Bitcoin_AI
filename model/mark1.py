import torch
import torch.nn as nn


def activation_function():
    return nn.PReLU()


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
        self.start_weight = nn.Parameter(torch.zeros(1, 12, 1))
        self.start_bias = nn.Parameter(torch.zeros(1, 12, 1))
        self.start_softmax = nn.Softmax(dim=1)

        self.net_start = nn.Conv1d(12, args.channel, args.kernal_size)
        
        self.input_size = args.input_size - args.kernal_size + 1
        input_size = self.input_size
        net_list = list()
        while input_size > 16:
            net_list.append(basic_block(args.channel))
            input_size //= 2
        net_list.append(basic_block(args.channel, False))
        self.net_list = nn.ModuleList(net_list)

        self.fc = nn.Sequential(
            nn.Linear(args.channel * 4 * len(net_list), args.channel),
            nn.Dropout(),
            activation_function(),
            nn.Linear(args.channel, 1, bias=False),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = (x - self.start_bias) * self.start_softmax(self.start_weight)
        x_large = self.net_start(x)

        input_size = self.input_size
        x_list = list()
        for net in self.net_list:
            x_list.append(net(x_large[:, :, -input_size:]))
            input_size //= 2

        x = torch.cat(x_list, dim=1).squeeze(2)
        x = self.fc(x).squeeze(1)
        return x

    def save_param(self, file_name='./param/param.pth'):
        torch.save(self.state_dict(), file_name)

    def load_param(self, file_name='./param/param.pth'):
        self.load_state_dict(torch.load(file_name, map_location=self.args.device))
