import argparse

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--issue',type=str, required=True, choices=['ele','solar','stock','traffic'], default='traffic')
parser.add_argument('--model', type=str, required=True, choices=['LSTNet','CNN','RNN','MHA_Net'] ,default='LSTNet')

parser.add_argument('--window', type=int, default=24 * 7)
parser.add_argument('--horizon', type=int, default=12)

parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units each layer')
parser.add_argument('--rnn_layers', type=int, default=1, help='number of RNN hidden layers')

parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units (channels)')
parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=24, help='The window size of the highway component')

parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)

parser.add_argument('--clip', type=float, default=10.,help='gradient clipping')
parser.add_argument('--epochs', type=int, default=200, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,help='random seed')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',help='report interval')

parser.add_argument('--cuda', type=str, default=True)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--amsgrad', type=str, default=True)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--skip', type=float, default=24)
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='sigmoid')