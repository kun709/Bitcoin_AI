def print_predict(open_, close_):
    s = 'open : '
    for o in open_:
        s += '{:1.3f}% '.format((o - 1) * 100)
    s += '| close : '
    for c in close_:
        s += '{:1.3f}% '.format((c - 1) * 100)
    print(s)
