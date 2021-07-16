import pybithumb
import datetime


orderbook = pybithumb.get_orderbook("BTC")

dt = datetime.datetime.fromtimestamp(int(orderbook["timestamp"])/1000)
print('timestamp :', dt)

bids = orderbook['bids']
print('\nbids')
for bid in bids:
    print(bid)

asks = orderbook['asks']
print('\nasks')
for ask in asks:
    print(ask)
