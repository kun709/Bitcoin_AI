import requests
import bs4
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import sqlite3
import ccxt

start = 1504224000000
start = 1582748490000

ep = 'https://api.binance.com'
candle = '/api/v3/klines'

start_date = []
start_date = []
start_time = []
open = []
high = []
low = []
close = []
volume = []

ticker = 'BTCUSDT'

first_params_candle = {'symbol': ticker, 'interval': '1m', 'startTime': start, 'limit': 1}
r1 = requests.get(ep+candle, params=first_params_candle)

# print(r1.text)


while len(r1.json()) >0:
    first_params_candle = {'symbol': 'BTCKRW', 'interval': '5m', 'startTime': start, 'limit': 1000}
    r1 = requests.get(ep+candle, params=first_params_candle)
    for i in range(0, len(r1.json())):
        print(datetime.fromtimestamp(r1.json()[i][0]/1000), r1.json()[i][1])
        start_date.append(datetime.fromtimestamp(r1.json()[i][0]/1000).strftime('%Y%m%d'))
        start_time.append(datetime.fromtimestamp(r1.json()[i][0]/1000).strftime('%H%M'))
        open.append(r1.json()[i][1])
        high.append(r1.json()[i][2])
        low.append(r1.json()[i][3])
        close.append(r1.json()[i][4])
        volume.append(r1.json()[i][5])
    if len(r1.json()) > 0:
        start = r1.json()[-1][6]+1
        print(datetime.fromtimestamp(r1.json()[-1][6]/1000).strftime('%Y%m%d'), '데이터 다운로드 완료')

    else:
        print('완료')
    time.sleep(1)

tickerline = [ticker]*len(start_date)
chart_data = {'date': start_date, 'time': start_time, 'ticker': ticker, 'open': open, 'high': high, 'low': low,
              'close': close, 'volume': volume}
df = pd.DataFrame( chart_data, columns=['date', 'time', 'ticker', 'open', 'high', 'low', 'close', 'volume'])

df.to_csv('larry.csv', header=True, index=False)
