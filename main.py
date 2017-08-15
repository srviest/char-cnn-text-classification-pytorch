#! /usr/bin/env python
import os
import argparse
import datetime
import torch
# import torchtext.data as data
# import torchtext.datasets as datasets
# import model_char_old
import model_char
import model
import train
# from data_loader_txt import mr
from data_loader_char import AGNEWs
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.005, help='initial learning rate [default: 0.01]')
parser.add_argument('-epochs', type=int, default=2500, help='number of epochs for train [default: 2500]')
parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 128]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
# data 
parser.add_argument('-train-path', metavar='DIR',
                    help='path to training data csv', default='data/ag_news_csv/train.csv')
parser.add_argument('-val-path', metavar='DIR',
                    help='path to validating data csv', default='data/ag_news_csv/test.csv')
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
parser.add_argument('-alphabet-path', default='alphabet.json', help='Contains all characters for prediction')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu' )
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()



# load data
# text_field = data.Field(lower=True)
# label_field = data.Field(sequential=False)
# train_iter, dev_iter = mr(text_field, label_field, batch_size=args.batch_size, device=-1, repeat=False)

print("\nLoading training data...")
train_dataset = AGNEWs(label_data_path=args.train_path, alphabet_path=args.alphabet_path)
print("\nTransferring training data into iterator...")
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)

print("\nLoading validating data...")
dev_dataset = AGNEWs(label_data_path=args.val_path, alphabet_path=args.alphabet_path)
print("\nTransferring validating data into iterator...")
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

#train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)


# update args and print
# args.embed_num = len(text_field.vocab)
# args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))


# model
if args.snapshot is None:
    num_features = len(train_dataset.alphabet)
    cnn = model_char.CharCNN(num_features)
    # cnn = model.CNN_Text(args)
    
else :
    print('\nLoading model from [%s]...' % args.snapshot)
    try:
        cnn = torch.load(args.snapshot)
    except :
        print("Sorry, This snapshot doesn't exist."); exit()

if args.cuda:
    cnn = cnn.cuda()
        

# train or predict
if args.predict is not None:
    label = train.predict(args.predict, cnn, text_field, label_field)
    print('\n[Text]  {}[Label] {}\n'.format(args.predict, label))
elif args.test :
    try:
        train.eval(test_iter, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else :
    print()
    train.train(train_loader, dev_loader, cnn, args)
    

