#! /usr/bin/env python
import os
import argparse
import datetime
import sys
import errno
from model import CharCNN
from data_loader import AGNEWs
from metric import print_f_score
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim
import torch.optim.lr_scheduler
import torch.autograd as autograd
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Character level CNN text classifier training')
# learning
learn = parser.add_argument_group('Learning options')
learn.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.0005]')
learn.add_argument('--epochs', type=int, default=200, help='number of epochs for train [default: 200]')
learn.add_argument('--batch-size', type=int, default=128, help='batch size for training [default: 128]')
learn.add_argument('--adaptive-lr', action='store_true', default=False, help='use dynamic learning schedule.')
# data 
parser.add_argument('--train-path', metavar='DIR',
                    help='path to training data csv', default='data/ag_news_csv/train.csv')
parser.add_argument('--val-path', metavar='DIR',
                    help='path to validating data csv', default='data/ag_news_csv/test.csv')
# model (text classifier)
cnn = parser.add_argument_group('Model options')
cnn.add_argument('--alphabet-path', default='alphabet.json', help='Contains all characters for prediction')
cnn.add_argument('--l0', type=int, default=1014, help='maximum length of input sequence to CNNs [default: 1014]')
cnn.add_argument('--shuffle', action='store_true', default=False, help='shuffle the data every epoch')
cnn.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
# device
device = parser.add_argument_group('Device options')
device.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
device.add_argument('--cuda', action='store_true', default=True, help='enable the gpu' )
# experiment options
experiment = parser.add_argument_group('Experiment options')
experiment.add_argument('--verbose', dest='verbose', action='store_true', default=False, help='Turn on progress tracking per iteration for debugging')
experiment.add_argument('--checkpoint', dest='checkpoint', default=True, action='store_true', help='Enables checkpoint saving of model')
experiment.add_argument('--save-folder', default='models_CharCNN_test/', help='Location to save epoch models, training configurations and results.')
experiment.add_argument('--log-config', default=True, action='store_true', help='Store experiment configuration')
experiment.add_argument('--log-result', default=True, action='store_true', help='Store experiment result')
experiment.add_argument('--log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
experiment.add_argument('--test-interval', type=int, default=200, help='how many steps to wait before vaidating [default: 200]')
experiment.add_argument('--save-interval', type=int, default=1, help='how many epochs to wait before saving [default:1]')


def train(train_loader, dev_loader, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.adaptive_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    model.train()

    for epoch in range(1, args.epochs+1):
        for i_batch, (data) in enumerate(train_loader):

            inputs, target = data            
            target.sub_(1)
        
            if args.cuda:
                inputs, target = inputs.cuda(), target.cuda()

            inputs = autograd.Variable(inputs)
            target = autograd.Variable(target)
            logit = model(inputs)
            loss = F.nll_loss(logit, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.verbose:
                print('\nTargets, Predicates')
                print(torch.cat((target.unsqueeze(1), torch.unsqueeze(torch.max(logit, 1)[1].view(target.size()).data, 1)), 1))
                print('\nLogit')
                print(logit)
            i_batch += 1
            if i_batch % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/args.batch_size
                sys.stdout.write(
                    '\rEpoch[{}] Batch[{}] - loss: {:.6f}  lr: {:.5f}  acc: {:.3f}%({}/{})'.format(epoch,
                                                                             i_batch,
                                                                             loss.data[0],
                                                                             optimizer.state_dict()['param_groups'][0]['lr'],
                                                                             accuracy,
                                                                             corrects,
                                                                             args.batch_size))
            if i_batch % args.test_interval == 0:
                val_loss = eval(dev_loader, model, epoch, i_batch, args)
                if args.adaptive_lr:
                    scheduler.step(val_loss)
                    clr = optimizer.state_dict()['param_groups'][0]['lr']
                    print('Reduce learning rate to: {lr:.6f}'.format(clr))
        if epoch % args.save_interval == 0:
            file_path = '%s/CharCNN_%d.pth.tar' % (args.save_folder, epoch)
            torch.save(model.state_dict(), file_path)

def eval(data_loader, model, epoch_train, batch_train, args):
    model.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
    predicates_all, target_all = [], []
    for i_batch, (data) in enumerate(data_loader):
        inputs, target = data
        target.sub_(1)

        if args.cuda:
            inputs, target = inputs.cuda(), target.cuda()

        inputs = autograd.Variable(inputs)
        target = autograd.Variable(target)
        logit = model(inputs)
        accumulated_loss += F.nll_loss(logit, target, size_average=False).data[0]
        predicates = torch.max(logit, 1)[1].view(target.size()).data
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        size+=len(target)
        predicates_all+=predicates.cpu().numpy().tolist()
        target_all+=target.data.cpu().numpy().tolist()

    avg_loss = accumulated_loss/size
    accuracy = 100.0 * corrects/size
    model.train()
    
    print('\nEvaluation - loss: {:.6f}  acc: {:.3f}%({}/{}) '.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    print_f_score(predicates_all, target_all)
    print('\n')

    if args.log_result:
        with open(os.path.join(args.save_folder,'result.csv'), 'a') as r:
            r.write('\n{:d},{:d},{:.5f},{:.2f},{:f}'.format(epoch_train, batch_train, avg_loss, accuracy, args.lr))

    return avg_loss

def main():
    # parse arguments
    args = parser.parse_args()

    # load training data
    print("\nLoading training data...")
    train_dataset = AGNEWs(label_data_path=args.train_path, alphabet_path=args.alphabet_path)
    print("Transferring training data into iterator...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, shuffle=True)
    # feature length
    args.num_features = len(train_dataset.alphabet)

    # load developing data
    print("\nLoading developing data...")
    dev_dataset = AGNEWs(label_data_path=args.val_path, alphabet_path=args.alphabet_path)
    print("Transferring developing data into iterator...")
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    
    # make save folder
    try:
        os.makedirs(args.save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise
    # args.save_folder = os.path.join(args.save_folder, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # configuration
    print("\nParameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}:".format(attr.capitalize().replace('_', ' ')).ljust(25)+"{}".format(value))

    # log result
    if args.log_result:
        with open(os.path.join(args.save_folder,'result.csv'), 'w') as r:
            r.write('{:s},{:s},{:s},{:s},{:s}'.format('epoch', 'batch', 'loss', 'acc', 'lr'))
    # model
    cnn = CharCNN(args)
    
    # using GPU
    if args.cuda:
        cnn = cnn.cuda()
            
    # train 
    train(train_loader, dev_loader, cnn, args)


if __name__ == '__main__':
    main()