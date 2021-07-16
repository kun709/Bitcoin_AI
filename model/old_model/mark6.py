import torch
import torch.nn as nn


def basic_lstm(input_size, args):
    return nn.LSTM(input_size=input_size,
                   hidden_size=args.lstm_hidden_size,
                   num_layers=args.lstm_num_layers,
                   proj_size=args.lstm_proj_size,
                   dropout=args.lstm_dropout,
                   batch_first=True)


class Mark(nn.Module):
    def __init__(self, args):
        super(Mark, self).__init__()
        self.args = args
        self.device = args.device

        self.net_256 = basic_lstm(5, args)
        self.net_128 = basic_lstm(args.lstm_proj_size, args)
        self.net_64 = basic_lstm(args.lstm_proj_size, args)
        self.net_32 = basic_lstm(args.lstm_proj_size, args)
        self.net_16 = basic_lstm(args.lstm_proj_size, args)

        self.fc = nn.Linear(args.lstm_proj_size * 5, args.predict_size * 2, bias=False)

    def forward(self, price, volume):
        price = (price + 1) / 2
        x = torch.cat([price, volume.unsqueeze(-1)], dim=2)

        x_256, _ = self.net_256(x)
        x_128, _ = self.net_128(x_256[:, 1::2])
        x_64, _ = self.net_64(x_128[:, 1::2])
        x_32, _ = self.net_32(x_64[:, 1::2])
        x_16, _ = self.net_16(x_32[:, 1::2])
        x = torch.cat([x_256[:, -1], x_128[:, -1], x_64[:, -1], x_32[:, -1], x_16[:, -1]], dim=1)
        x = self.fc(x).view(-1, 2, self.args.predict_size)
        return x

    def save_param(self, file_name='./param/param.pth'):
        torch.save(self.state_dict(), file_name)

    def load_param(self, file_name='./param/param.pth'):
        self.load_state_dict(torch.load(file_name, map_location=self.args.device))
