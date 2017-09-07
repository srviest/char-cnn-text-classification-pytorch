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

parser = argparse.ArgumentParser(description='Character level CNN text classifier testing', formatter_class=argparse.RawTextHelpFormatter)
# model
parser.add_argument('--model-path', default=None, help='Path to pre-trained acouctics model created by DeepSpeech training')
# data
parser.add_argument('--test-path', metavar='DIR',
                    help='path to testing data csv', default='data/ag_news_csv/test.csv')
parser.add_argument('--batch-size', type=int, default=20, help='batch size for training [default: 128]')
parser.add_argument('-alphabet-path', default='alphabet.json', help='Contains all characters for prediction')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-cuda', action='store_true', default=True, help='enable the gpu' )
# logging options
parser.add_argument('-verbose', dest='verbose', action='store_true', default=False, help='Turn on progress tracking per iteration for debugging')
parser.add_argument('-save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
args = parser.parse_args()


if __name__ == '__main__':


    # load testing data
    print("\nLoading testing data...")
    test_dataset = AGNEWs(label_data_path=args.test_path, alphabet_path=args.alphabet_path)
    print("Transferring testing data to iterator...")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    _, num_class_test = test_dataset.get_class_weight()
    print('\nNumber of testing samples: '+str(test_dataset.__len__()))
    for i, c in enumerate(num_class_test):
        print("\tLabel {:d}:".format(i).ljust(15)+"{:d}".format(c).rjust(8))

    args.num_features = len(test_dataset.alphabet)
    model = CharCNN(args)
    print("=> loading weights from '{}'".format(args.model_path))
    assert os.path.isfile(args.model_path), "=> no checkpoint found at '{}'".format(args.model_path)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])

    # using GPU
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    model.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
    predicates_all, target_all = [], []
    print('\nTesting...')
    for i_batch, (data) in enumerate(test_loader):
        inputs, target = data
        size+=len(target)
        if args.cuda:
            inputs, target = inputs.cuda(), target.cuda()

        inputs = Variable(inputs, volatile=True)
        target = Variable(target)
        logit = model(inputs)
        predicates = torch.max(logit, 1)[1].view(target.size()).data
        accumulated_loss += F.nll_loss(logit, target, size_average=False).data[0]
        corrects += (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        predicates_all+=predicates.cpu().numpy().tolist()
        target_all+=target.data.cpu().numpy().tolist()
        
    avg_loss = accumulated_loss/size
    accuracy = 100.0 * corrects/size
    print('\rEvaluation - loss: {:.6f}  acc: {:.3f}%({}/{}) '.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    print_f_score(predicates_all, target_all)