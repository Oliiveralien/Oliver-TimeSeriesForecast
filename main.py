import math
import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import importlib
import numpy as np
from utils import Data_Utility
from train_eval import train, evaluate, makeOptimizer

import args_file
args = args_file.parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from logger import Logger
from models import LSTNet,MHA_Net,CNN,RNN
import matplotlib.pyplot as plt
plt.switch_backend('agg')


save_root = './save_models/extra'
save_path = os.path.join(save_root, args.issue, args.model+'.pt')
if os.path.exists(os.path.join(save_root, args.issue)) is False:
    os.makedirs(os.path.join(save_root, args.issue))

data_root = './data'
data_path = os.path.join(data_root, args.issue+'.txt')

log_root = './logs/extra'
log_path = os.path.join(log_root, args.issue, args.model)
if os.path.exists(log_path) is False:
    os.makedirs(log_path)

plt_title = 'Evaluation on '+args.issue+' of '+args.model
plt_root = './evaluation_pics/extra'
plt_path = os.path.join(plt_root, args.issue, args.model+'.jpg')
if os.path.exists(os.path.join(plt_root, args.issue)) is False:
    os.makedirs(os.path.join(plt_root, args.issue))

csv_root = './csv_files/extra'
csv_path = os.path.join(csv_root, args.issue+'.csv')

# Choose device: cpu or gpu
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility.
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
Data = Data_Utility(data_path, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalize)

# loss function
if args.L1Loss:
    criterion = nn.L1Loss(size_average=False)
else:
    criterion = nn.MSELoss(size_average=False)
evaluateL2 = nn.MSELoss(size_average=False)
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda()
    evaluateL2 = evaluateL2.cuda()

# Select model
model = eval(args.model).Model(args, Data)
train_method = train
eval_method = evaluate
nParams = sum([p.nelement() for p in model.parameters()])
print('number of parameters: %d' % nParams)
if args.cuda:
    model = nn.DataParallel(model)

# print(Data.train[0].size())
# import tensorwatch as tw
# img = tw.draw_model(model,[1,168,321])
# img.save(r'./CNN_net.png')
# print('done')


best_val = 10000000

optim = makeOptimizer(model.parameters(), args)

val_logger = Logger(os.path.join(log_path, 'train.log'),
                          ['epoch', 'time used', 'train_loss', 'valid rse', 'valid rae', 'valid corr'])

test_logger = Logger(os.path.join(log_path, 'val.log'), ['epoch', 'test rse', 'test rae', 'test corr'])

# While training you can press Ctrl + C to stop it.
try:
    print('Training start')

    ax = []
    ay = []
    val = []
    plt.ion()
    plt.title(plt_title)
    plt.xlabel("epoch")
    plt.ylabel("val_corr")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train_loss = train_method(Data, Data.train[0], Data.train[1], model, criterion, optim, args)

        val_loss, val_rae, val_corr = eval_method(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1, args)
        print('| end of epoch {:3d} | time used: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.
                format( epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))

        val.append(val_corr)

        val_logger.log({
            'epoch': '{:3d}'.format(epoch),
            'time used': '{:5.2f}s'.format(time.time() - epoch_start_time),
            'train_loss': '{:5.4f}'.format(train_loss),
            'valid rse': '{:5.4f}'.format(val_loss),
            'valid rae': '{:5.4f}'.format(val_rae),
            'valid corr': '{:5.4f}'.format(val_corr)
        })

        ax.append(epoch)
        ay.append(val_corr)
        plt.plot(ax,ay,'b.--',linewidth=1)

        if val_loss < best_val:
            with open(save_path, 'wb') as f:
                torch.save(model, f)
            best_val = val_loss
        if epoch % 10 == 0:
            test_loss, test_rae, test_corr = eval_method(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,args)
            print("| test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}\n".format(test_loss, test_rae, test_corr))

            test_logger.log({
            'epoch': epoch // 10,
            'test rse': '{:5.4f}'.format(test_loss),
            'test rae': '{:5.4f}'.format(test_rae),
            'test corr': '{:5.4f}'.format(test_corr)
        })
    plt.ioff()
    plt.savefig(plt_path)
    with open(csv_path,'ab') as f:
        np.savetxt(f, val, delimiter = ',', fmt='%.4f')

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(save_path, 'rb') as f:
    model = torch.load(f)
test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, evaluateL2, evaluateL1,args)
print('Best model performance:')
print("| test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
