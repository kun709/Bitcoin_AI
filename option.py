import argparse


def option():
    parser = argparse.ArgumentParser(description='난 비트코인 부자가 될거야')

    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--online-epochs', type=int, default=1, metavar='N',
                        help='number of epochs to online train (default: 14)')

    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--input-size', type=int, default=(256 + 2))
    parser.add_argument('--predict-size', type=int, default=6)

    parser.add_argument('--kernal_size', type=int, default=3)
    parser.add_argument('--channel', type=int, default=64)
    parser.add_argument('--atoms', type=int, default=11)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--lstm_hidden_size', type=int, default=128)
    parser.add_argument('--lstm_proj', type=bool, default=True)
    parser.add_argument('--lstm_proj_size', type=int, default=64)
    parser.add_argument('--lstm_num_layers', type=int, default=2)
    parser.add_argument('--lstm_dropout', type=float, default=0.2)

    parser.add_argument('--volume-a', type=float, default=0.5014)
    parser.add_argument('--volume-b', type=float, default=1.454)
    parser.add_argument('--price-a', type=list, default=[1.0, 1.0015509, 0.99838033, 1.0])
    parser.add_argument('--price-b', type=list, default=[1000, 146.7, 132.8, 124.4])

    parser.add_argument('--root', type=str, default='./data')

    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser.parse_args()
