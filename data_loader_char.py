import csv
import re
import torch
import codecs
import json
from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd

class AGNEWs(Dataset):
    def __init__(self, label_data_path, alphabet_path, l0=1014):
        """Create AG's News dataset object.

        Arguments:
            label_data_path: The path of label and data file in csv.
            l0: max length of a sample.
            alphabet_path: The path of alphabet json file.
        """
        self.label_data_path = label_data_path
        # read alphabet
        with open(alphabet_path) as alphabet_file:
            alphabet = str(''.join(json.load(alphabet_file)))
        self.alphabet = alphabet
        self.l0 = l0
        self.label, self.data = self.load()

        self.y = torch.LongTensor(self.label)
        self.X = self.oneHotEncode(self.data)
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        
        sample = {'label': self.y[idx], 'data': self.X[idx]}
        return sample

    def load(self, lowercase=True):
        label = []
        data = []
        with open(self.label_data_path, 'rb') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            # num_samples = sum(1 for row in rdr)
            for index, row in enumerate(rdr):
                txt = ""
                for s in row[1:]:
                    txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
                label.append(int(row[0]))
                if lowercase:
                    txt = txt.lower()
                data.append(txt)
        return label, data

    def oneHotEncode(self, data):
        X = torch.zeros(len(data), len(self.alphabet), self.l0)
        for index_seq, sequence in enumerate(data):
            for index_char, char in enumerate(sequence):
                if self.char2Index(char)!=-1:
                    X[index_seq][self.char2Index(char)][index_char] = 1
        return X

    def char2Index(self, character):

        return self.alphabet.find(character)


if __name__ == '__main__':
    
    label_data_path = '/Users/ychen/Documents/TextClfy/data/ag_news_csv/test.csv'
    alphabet_path = '/Users/ychen/Documents/TextClfy/alphabet.json'

    train_dataset = AGNEWs(label_data_path, alphabet_path)
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=4)


    for i_batch, sample_batched in enumerate(train_loader):
        print(sample_batched['label'].size())
        target = sample_batched['label']
        print('type(target): ', target)
        target = target.float()
        print('type(target): ', target)
        # inputs = autograd.Variable(inputs)
        # print(inputs.data)
        # print(sample_batched['data'][0])
        # print(sample_batched['label'])
        # print i_batch
        # observe 4th batch and stop.
        if i_batch == 0:
            break

    print(len(train_loader))
    