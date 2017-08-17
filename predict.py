import os
import argparse
import datetime
import sys
import errno
import model_CharCNN
from data_loader import AGNEWs
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.autograd as autograd
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Character level CNN text classifier inference')
# data 
parser.add_argument('-val-path', metavar='DIR',
                    help='path to validating data csv', default='data/ag_news_csv/test.csv')
parser.add_argument('-alphabet-path', default='alphabet.json', help='Contains all characters for prediction')

# device

parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu' )
# logging options
parser.add_argument('-verbose', dest='verbose', action='store_true', default=False, help='Turn on progress tracking per iteration for debugging')
parser.add_argument('-checkpoint', dest='checkpoint', default=True, action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('-save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before vaidating [default: 100]')
parser.add_argument('-save-interval', type=int, default=20, help='how many epochs to wait before saving [default:10]')


else :
        print('\nLoading model from [%s]...' % args.snapshot)
        try:
            cnn = torch.load(args.snapshot)
        except :
            print("Sorry, This snapshot doesn't exist."); exit()


if args.predict is not None:
        label = train.predict(args.predict, cnn, text_field, label_field)
        print('\n[Text]  {}[Label] {}\n'.format(args.predict, label))




def eval(data_loader, model, args):
    model.eval()
    corrects, avg_loss, size = 0, 0, 0
    # criterion = nn.NLLLoss(size_average=False)
    # for batch in data_loader:
    for i_batch, sample_batched in enumerate(data_loader):
        inputs = sample_batched['data']
        target = sample_batched['label']
        target.sub_(1)
        # inputs, target = batch.text, batch.label
        # inputs.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            inputs, target = inputs.cuda(), target.cuda()

        inputs = autograd.Variable(inputs)
        target = autograd.Variable(target)
        logit = model(inputs)
        # loss = F.cross_entropy(logit, target, size_average=False)
        # loss = criterion(logit, target)
        loss = F.nll_loss(logit, target, size_average=False)

        correct = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        batch_loss = loss.data[0]
        # print('correct: ', correct)
        # print('batch_loss: ', batch_loss)
        avg_loss += batch_loss
        corrects += correct
        size+=len(target)

    # print(len(target))
    # size = len(data_loader)
    avg_loss = loss.data[0]/size
    accuracy = 100.0 * corrects/size
    # print('loss.data[0]: ', loss.data[0])
    # print('corrects: ', corrects)
    # print('size: ', size)
    # print('avg_loss: ', avg_loss)
    # print('accuracy: ', accuracy)
    model.train()
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))


def predict(text, model, text_field, label_feild):
    assert isinstance(text, str)
    model.eval()
    text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0][0]+1]
