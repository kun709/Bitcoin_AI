import pybithumb
import numpy as np
import torch
from option import option


class RealTimeData:
    def __init__(self, args):
        self.price_gap = args.price_gap
        self.input_size = args.input_size

    def get_data(self):
        btc = pybithumb.get_ohlcv("BTC", interval="minute5")

        volume = np.clip(np.asarray(btc.iloc[:, 4]), 1e-4, 1e4)
        volume_data = torch.from_numpy(1 - (1 / (1 + (volume[1:] / volume[:-1])))).float()

        price = np.asarray(btc.iloc[:, :4])
        price_data = torch.from_numpy(np.clip((price[:, 1:] / price[:, [0]] - 1) / self.price_gap, -1, 1)).float()

        price_24h = price_data[-self.input_size:].unsqueeze(0)
        volume_24h = volume_data[-self.input_size:].unsqueeze(0)

        return price_24h, volume_24h


if __name__ == "__main__":
    args = option()
    real_time_data = RealTimeData(args)
    print(real_time_data.get_data())

