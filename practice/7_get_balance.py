import pybithumb

con_key = "ef81b58f504c67e3eac280ea25d0e87d"
sec_key = "4a4b58b21a4759d03bee37b85f6a87f7"

bithumb = pybithumb.Bithumb(con_key, sec_key)
balance = bithumb.get_balance("BTC")
# 비트코인의 총 잔고, 거래 중인 비트코인의 수량, 보유 중인 총원화, 주문에 사용된 원화
print(balance)
