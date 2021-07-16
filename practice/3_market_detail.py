# ch05/05_05.py
import pybithumb

# 24시간 : 시가 / 고가 / 저가 / 종가 / 거래량
detail = pybithumb.get_market_detail("BTC")
print(detail)
