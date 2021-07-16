import pybithumb
import numpy as np
import torch.optim as optim
import torch
from tqdm import tqdm
from model.mark1 import Mark1
from loss import DistributeTrainLoss
from dataset import get_online_data_loader


class Agent:
    def __init__(self, args, file_name='./param/online.pth'):
        self.network = Mark1(args).to(args.device)
        self.network.load_param(file_name)

        self.args = args
        self.price_a = args.price_a
        self.price_b = args.price_b
        self.volume_a = args.volume_a
        self.volume_b = args.volume_b
        self.input_size = args.input_size
        self.device = args.device
        self.predict_size = args.predict_size
        con_key = "ef81b58f504c67e3eac280ea25d0e87d"
        sec_key = "4a4b58b21a4759d03bee37b85f6a87f7"
        self.bithumb = pybithumb.Bithumb(con_key, sec_key)

        self.linspace = torch.linspace(-1, 1, args.atoms).view(1, -1).to(args.device)
        self.last_price = self.get_current_price()

    def predict_price(self):
        self.network.eval()
        last_time, price, volume = self.get_data()
        price, volume = price.to(self.device), volume.to(self.device)

        output = self.network(price, volume)
        open_, high_, low_, close_ = self.decode_output(output)
        return last_time, open_, close_

    def decode_output(self, output):
        predict_exp = torch.exp(output)
        predict_distribute = predict_exp / predict_exp.sum(-1).unsqueeze(-1)
        predict_value = (self.linspace * predict_distribute).sum(-1).cpu().detach().numpy()

        open_ = predict_value[:, :, :, 0] / self.price_b[0] + self.price_a[0]
        high_ = predict_value[:, :, :, 1] / self.price_b[1] + self.price_a[1]
        low_ = predict_value[:, :, :, 2] / self.price_b[2] + self.price_a[2]
        close_ = predict_value[:, :, :, 3] / self.price_b[3] + self.price_a[3]

        return open_, high_, low_, close_

    def strategy(self, open, close, bought, n=3):
        price = 1
        for i in range(1, n):
            price *= open[i] * close[i]
        if not bought:
            if price > 1:
                return 'BUY'
        else:
            if price < 1:
                return 'SELL'
        return None

    def get_current_price(self, ticker='BTC'):
        return pybithumb.get_current_price(ticker)

    def get_data(self):
        btc = pybithumb.get_ohlcv("BTC", interval="minute5")

        last_time = btc.index[-1]
        if last_time.minute % 5 != 0:
            volume = np.asarray(btc.iloc[:-1, 4]) + 1e-8
            price = np.asarray(btc.iloc[:-1, :4])
            last_time = btc.index[-2]
        else:
            volume = np.asarray(btc.iloc[:, 4]) + 1e-8
            price = np.asarray(btc.iloc[:, :4])

        volume = (0.5 - (1 / (1 + (volume[1:] / volume[:-1])))) * self.volume_b + self.volume_a
        price = (price[1:] / price[:-1, [3]] - np.array([self.price_a])) * np.array([self.price_b])

        volume_data = torch.from_numpy(np.clip(volume, 0, 1)).float()
        price_data = torch.from_numpy(np.clip(price, -1, 1)).float()

        price_24h = price_data[-self.input_size:].unsqueeze(0)
        volume_24h = volume_data[-self.input_size:].unsqueeze(0)

        return last_time, price_24h, volume_24h

    def get_balance(self, coin='BTC'):
        return self.bithumb.get_balance(coin)

    def buy_coin(self, amount, coin='BTC'):
        sell_price = pybithumb.get_orderbook(coin)['asks'][0]['price']
        if amount < sell_price * 0.001:
            return None
        unit = amount / float(sell_price)
        order_id = self.bithumb.buy_market_order(coin, unit)
        return order_id

    def sell_coin(self, amount, coin='BTC'):
        if amount < 0:
            unit = self.bithumb.get_balance("BTC")[0]
            order_id = self.bithumb.sell_market_order("BTC", unit)
            return order_id
        else:
            sell_price = pybithumb.get_orderbook(coin)['bids'][0]['price']
            unit = amount / float(sell_price)
            order_id = self.bithumb.sell_market_order("BTC", unit)
            return order_id

    def online_train(self, file_name='./param/online.pth'):
        self.network.train()
        train_loader = get_online_data_loader(self.args, self.input_size+self.predict_size)
        optimizer = optim.Adam(self.network.parameters(), lr=1e-5, weight_decay=1e-5)
        loss_func = DistributeTrainLoss(self.args)
        for epoch in range(1, self.args.online_epochs + 1):
            total_loss = 0
            for data in train_loader:
                price, volume = data
                price, volume = price.to(self.device), volume.to(self.device)
                optimizer.zero_grad()
                output = self.network(price[:, :args.input_size], volume[:, :self.input_size])
                loss = loss_func(output, price) + self.network.ipe_loss() * 1e-4
                loss.backward()
                optimizer.step()

            self.network.save_param(file_name)
        print('online train loss : {:2.5f}'.format(total_loss / len(train_loader)))


if __name__ == "__main__":
    from option import option
    args = option()
    real_time_data = Agent(args, 'online.pth')
    print(real_time_data.predict_price())
    print(real_time_data.get_balance())
    print(real_time_data.online_train('online.pth'))


