import torch
import torch.nn as nn


class BitCoinNetwork(nn.Module):
    def __init__(self, args):
        super(BitCoinNetwork, self).__init__(args)
        self.args = args

    def save_param(self, file_name='param.pth'):
        torch.save(self.state_dict(), self.args.param_path + file_name)

    def load_param(self, file_name='param.pth'):
        self.load_state_dict(torch.load(self.args.param_path + file_name, map_location=self.args.device))
