#!/usr/bin/env python3
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

if __name__ == '__main__':
    model = DeepSpeech.load_model(args.model_path, cuda=args.cuda)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    if args.decoder == "beam":
        scorer = None
        if args.lm_path is not None:
            scorer = KenLMScorer(labels, args.lm_path, args.trie_path)
            scorer.set_lm_weight(args.lm_alpha)
            scorer.set_word_weight(args.lm_beta1)
            scorer.set_valid_word_weight(args.lm_beta2)
        else:
            scorer = Scorer()
        decoder = BeamCTCDecoder(labels, scorer, beam_width=args.beam_width, top_paths=1, space_index=labels.index(' '), blank_index=labels.index('_'))
    else:
        decoder = GreedyDecoder(labels, space_index=labels.index(' '), blank_index=labels.index('\''))

    audio_paths = []
    if os.path.isdir(args.audio_path):
        audio_paths = glob.glob(args.audio_path+os.sep+'*.wav')
    else:
        audio_paths.append(args.audio_path)

    parser = SpectrogramParser(audio_conf, normalize=True)

    for audio_path in audio_paths:
        t0 = time.time()
        spect = parser.parse_audio(audio_path).contiguous()
        spect = spect.view(1, 1, spect.size(0), spect.size(1))
        out = model(Variable(spect, volatile=True))
        out = out.transpose(0, 1)  # TxNxH

        if args.prob:
            out_numpy = out.data.cpu().numpy()
            t1 = time.time()
            print(out_numpy)
        else:
            decoded_output = decoder.decode(out.data)
            t1 = time.time()
            print(decoded_output[0])
            
        print("Decoded {0:.2f} seconds of audio in {1:.2f} seconds\n".format(spect.size(3)*audio_conf['window_stride'], t1-t0))