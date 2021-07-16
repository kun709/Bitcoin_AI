import pybithumb


price = pybithumb.get_current_price("BTC")
print(price)

detail = pybithumb.get_market_detail("BTC")
print(detail)
