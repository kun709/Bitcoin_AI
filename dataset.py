from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
import pybithumb


class DataSet(Dataset):
    def __init__(self, args, data_size,  train=True):
        self.data_size = data_size
        self.train = train
        self.args = args
        self.csv_data = pd.read_csv('./data/larry.csv')

        volume = np.asarray(self.csv_data.iloc[:, 7]) + 1e-8
        volume = (0.5 - (1 / (1 + (volume[1:] / volume[:-1])))) * args.volume_b + args.volume_a
        self.volume_data = torch.from_numpy(np.clip(volume, 0, 1)).float()

        price = np.asarray(self.csv_data.iloc[1:, 3:7])
        price = (price[1:] / price[:-1, [3]] - np.array([args.price_a])) * np.array([args.price_b])
        self.price_data = torch.from_numpy(np.clip(price, -1, 1)).float()

        self.data_len = self.price_data.size(0) - (self.data_size + 1)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.price_data[idx:idx + self.data_size], self.volume_data[idx:idx + self.data_size]


class OnlineDataSet(Dataset):
    def __init__(self, args, data_size):
        self.data_size = data_size
        self.args = args
        btc = pybithumb.get_ohlcv("BTC", interval="minute5")
        self.data_len = len(btc) - (self.data_size + 1)
        if btc.index[-1].minute % 5 != 0:
            volume = np.asarray(btc.iloc[:-1, 4]) + 1e-8
            price = np.asarray(btc.iloc[:-1, :4])
            self.data_len -= 1
        else:
            volume = np.asarray(btc.iloc[:, 4]) + 1e-8
            price = np.asarray(btc.iloc[:, :4])

        volume = (0.5 - (1 / (1 + (volume[1:] / volume[:-1])))) * args.volume_b + args.volume_a
        self.volume_data = torch.from_numpy(np.clip(volume, 0, 1)).float()
        price = (price[1:] / price[:-1, [3]] - np.array([args.price_a])) * np.array([args.price_b])
        self.price_data = torch.from_numpy(np.clip(price, -1, 1)).float()

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        return self.price_data[idx:idx + self.data_size], self.volume_data[idx:idx + self.data_size]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def data_clean(price: np.ndarray, volume: np.ndarray):
    # open high low close
    volume += 1e-8
    open__last_close = price[1:, 0] / price[:-1, 3]
    high__close = price[:, 1] / price[:, 3]
    low__close = price[:, 2] / price[:, 3]
    close__last_close = price[1:, 3] / price[:-1, 3]
    # volume__last_volume = volume[1:] / volume[:-1]

    close__ma5_close = price[4:, 3] / moving_average(price[:, 3], 5)
    close__ma10_close = price[9:, 3] / moving_average(price[:, 3], 10)
    close__ma20_close = price[19:, 3] / moving_average(price[:, 3], 20)
    close__ma60_close = price[59:, 3] / moving_average(price[:, 3], 60)

    volume__ma5_volume = volume[4:] / moving_average(volume, 5)
    volume__ma10_volume = volume[9:] / moving_average(volume, 10)
    volume__ma20_volume = volume[19:] / moving_average(volume, 20)
    volume__ma60_volume = volume[59:] / moving_average(volume, 60)

    data_list = [
        open__last_close[58:], high__close[59:], low__close[59:], close__last_close[58:],
        close__ma5_close[55:], close__ma10_close[50:], close__ma20_close[40:], close__ma60_close,
        volume__ma5_volume[55:], volume__ma10_volume[50:], volume__ma20_volume[40:], volume__ma60_volume
    ]

    result = torch.from_numpy(np.stack(data_list, axis=1)).float()
    return result


class ImproveDataSet(Dataset):
    def __init__(self, args,  train=True):
        self.input_size = args.input_size
        self.predict_size = args.predict_size
        self.train = train
        self.args = args
        self.csv_data = pd.read_csv('./data/larry.csv')

        volume = np.asarray(self.csv_data.iloc[:, 7])
        price = np.asarray(self.csv_data.iloc[:, 3:7])
        self.data = data_clean(price, volume)
        self.data_len = len(self.data) - (args.input_size + args.predict_size)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data_input = self.data[idx:idx + self.input_size]

        data_target = self.data[idx + self.input_size + 1:idx + self.input_size + self.predict_size, 3]
        data_target = (data_target.log().sum() > 0).long()
        return data_input, data_target


class ImproveOnlineDataSet(Dataset):
    def __init__(self, args):
        self.input_size = args.input_size
        self.predict_size = args.predict_size
        self.args = args
        btc = pybithumb.get_ohlcv("BTC", interval="minute5")
        if btc.index[-1].minute - btc.index[-2].minute in [5, -55]:
            volume = np.asarray(btc.iloc[:-1, 4])
            price = np.asarray(btc.iloc[:-1, :4])
        else:
            volume = np.asarray(btc.iloc[:, 4])
            price = np.asarray(btc.iloc[:, :4])

        self.data = data_clean(price, volume)
        self.data_len = len(self.data) - (args.input_size + args.predict_size)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        data_input = self.data[idx:idx + self.input_size]

        data_target = self.data[idx + self.input_size + 1:idx + self.input_size + self.predict_size, 3]
        data_target = (data_target.log().sum() > 0).long()
        return data_input, data_target


def get_data_loader(args):
    return DataLoader(ImproveDataSet(args), batch_size=args.batch_size, shuffle=True)


def get_online_data_loader(args):
    return DataLoader(ImproveOnlineDataSet(args), batch_size=args.batch_size, shuffle=True)


if __name__ == "__main__":
    from option import option
    args = option()
    data_set = ImproveDataSet(args, args.input_size+args.predict_size)
    print(data_set[0])
