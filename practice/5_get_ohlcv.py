import pybithumb
import pyupbit
'''
"day": "24h",
"hour12": "12h",
"hour6": "6h",
"hour": "1h",
"minute30": "30m",
"minute10": "10m",
"minute5": "5m",
"minute3": "3m",
"minute1": "1m",
'''

btc = pybithumb.get_ohlcv("BTC", interval="minute5")
print(btc)
print(type(btc))

btc_upbit = pyupbit.get_ohlcv('KRW-BTC', interval='minute5', count=100)
print(btc_upbit)
