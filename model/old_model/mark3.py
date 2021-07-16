import torch
import torch.nn as nn
from .IPE import InterpolePositionalEmbedding


class Mark(nn.Module):
    def __init__(self, args):
        super(Mark, self).__init__()
        self.args = args
        self.predict_size = args.predict_size
        self.atoms = args.atoms

        self.price_open_embed = InterpolePositionalEmbedding(args.lstm_input_size, args.atoms)
        self.price_low_embed = InterpolePositionalEmbedding(args.lstm_input_size, args.atoms)
        self.price_high_embed = InterpolePositionalEmbedding(args.lstm_input_size, args.atoms)
        self.price_close_embed = InterpolePositionalEmbedding(args.lstm_input_size, args.atoms)
        self.volume_embed = InterpolePositionalEmbedding(args.lstm_input_size, args.atoms)
        if args.lstm_proj:
            self.lstm_layer = nn.LSTM(input_size=args.lstm_input_size,
                                      hidden_size=args.lstm_hidden_size,
                                      num_layers=args.lstm_num_layers,
                                      proj_size=args.lstm_proj_size,
                                      dropout=args.lstm_dropout,
                                      batch_first=True)
        else:
            self.lstm_layer = nn.LSTM(input_size=args.lstm_input_size,
                                      hidden_size=args.lstm_hidden_size,
                                      num_layers=args.lstm_num_layers,
                                      dropout=args.lstm_dropout,
                                      batch_first=True)
        self.fc = nn.Linear(args.lstm_proj_size, self.predict_size * self.atoms * 4)

    def forward(self, price, volume):
        price = (price + 1) / 2
        open_ = self.price_open_embed(price[:, :, 0])
        high_ = self.price_low_embed(price[:, :, 1])
        low_ = self.price_high_embed(price[:, :, 2])
        close_ = self.price_close_embed(price[:, :, 3])
        volume = self.volume_embed(volume)
        x = open_ + low_ + high_ + close_ + volume
        # x = torch.cat([price, volume.unsqueeze(-1)], dim=2)

        x = self.lstm_layer(x)[0]
        x_size = x.size()
        x = self.fc(x).view(x_size[0], x_size[1], self.predict_size, 4, self.atoms)
        return x

    def ipe_loss(self):
        return self.price_open_embed.loss() + self.price_low_embed.loss() + self.price_high_embed.loss() + self.price_close_embed.loss() + self.volume_embed.loss()

    def save_param(self, file_name='./param/param.pth'):
        torch.save(self.state_dict(), file_name)

    def load_param(self, file_name='./param/param.pth'):
        self.load_state_dict(torch.load(file_name, map_location=self.args.device))
