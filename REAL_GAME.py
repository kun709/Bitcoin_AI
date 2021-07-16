from model import agent
from option import option
import time
from util import print_predict

args = option()
bitcoin_agent = agent.Agent(args, './param/online.pth')
last_time, price_24h, _ = bitcoin_agent.get_data()
flag = False
bought_flag = False

money = 200_000
coin = 0

while True:
    if flag:
        if last_time.minute == bitcoin_agent.get_data()[0].minute:
            print('.', end='')
            time.sleep(10)
            continue
    flag = True
    tm = time.localtime()
    string = time.strftime('%Y-%m-%d %I:%M:%S %p', tm)
    print('\n', string)
    last_time, open, close = bitcoin_agent.predict_price()
    print_predict(open, close)
    print('data time :', last_time.minute)

    s = bitcoin_agent.strategy(open, close, bought_flag, n=2)
    price = bitcoin_agent.get_current_price()
    if s is not None:
        if (not bought_flag) and (s == 'BUY'):
            coin = int(1000 * money / price) / 1000
            money -= coin * price
            bought_flag = True
            print('BUY', end=' ')
        elif bought_flag and (s == 'SELL'):
            money += coin * price
            coin = 0
            bought_flag = False
            print('SELL', end=' ')
    print('money:{:.0f} coin:{:.3f} value:{:.0f} current_price:{:.0f}'.format(money, coin, price * coin + money, price))

    bitcoin_agent.online_train()

    time.sleep(60)
    while time.localtime().tm_min % 5 != 0:
        time.sleep(5)




