import torch
import torch.nn as nn
from IPE import InterpolePositionalEmbedding


def downsample(channel, expansion=2, stride=4):
    return nn.Sequential(
        nn.Conv1d(channel, channel * expansion, kernel_size=stride, stride=stride),
        nn.BatchNorm1d(channel * expansion)
    )


class Conv1DBlock(nn.Module):
    def __init__(self, channel, kernal=3, padding=1):
        super(Conv1DBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channel, channel, kernal, padding=padding, bias=False),
            nn.BatchNorm1d(channel),
            nn.PReLU(),
            nn.Conv1d(channel, channel, kernal, padding=padding, bias=False),
            nn.BatchNorm1d(channel),
            nn.PReLU()
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(x + self.net(x))


class Mark(nn.Module):
    def __init__(self, args):
        super(Mark, self).__init__()
        self.args = args

        self.price_open_embed = InterpolePositionalEmbedding(args.channel, args.atoms)
        self.price_low_embed = InterpolePositionalEmbedding(args.channel, args.atoms)
        self.price_high_embed = InterpolePositionalEmbedding(args.channel, args.atoms)
        self.price_close_embed = InterpolePositionalEmbedding(args.channel, args.atoms)
        self.volume_embed = InterpolePositionalEmbedding(args.channel, args.atoms)

        self.net_start = nn.Sequential(
            nn.Conv1d(args.channel, args.channel, args.kernal_size, bias=False),
            nn.BatchNorm1d(args.channel),
            nn.PReLU()
        )

        self.net = nn.Sequential(
            Conv1DBlock(args.channel),
            Conv1DBlock(args.channel),
            downsample(args.channel),
            Conv1DBlock(args.channel * 2),
            Conv1DBlock(args.channel * 2),
            downsample(args.channel * 2, expansion=1)
        )
        self.rnn = nn.RNN(input_size=args.channel * 2, hidden_size=args.channel * 2, batch_first=True)

        linear_input_size = args.channel * 2 * ((args.input_size - 4) // 16)

        self.fc = nn.Sequential(
            nn.Linear(linear_input_size, args.atoms * args.predict_size * 8),
            nn.PReLU(),
            nn.Linear(args.atoms * args.predict_size * 8, args.atoms * args.predict_size * 4)
        )

    def forward(self, price, volume):
        price = (price + 1) / 2
        open_ = self.price_open_embed(price[:, :, 0])
        high_ = self.price_low_embed(price[:, :, 1])
        low_ = self.price_high_embed(price[:, :, 2])
        close_ = self.price_close_embed(price[:, :, 3])
        volume = self.volume_embed(volume)
        x = (open_ + low_ + high_ + close_ + volume).permute(0, 2, 1)

        x = self.net_start(x)
        x = self.net(x)

        x = self.rnn(x.permute(0, 2, 1))[0]

        x = torch.flatten(x, 1)
        x = self.fc(x).view(-1, 4 * self.args.predict_size, self.args.atoms)

        return x

    def ipe_loss(self):
        return self.price_open_embed.loss() + self.price_low_embed.loss() + self.price_high_embed.loss() + self.price_close_embed.loss() + self.volume_embed.loss()

    def save_param(self, file_name='./param/param.pth'):
        torch.save(self.state_dict(), file_name)

    def load_param(self, file_name='./param/param.pth'):
        self.load_state_dict(torch.load(file_name, map_location=self.args.device))
