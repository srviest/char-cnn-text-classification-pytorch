## Introduction
This is the implementation of Zhang's [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626) paper in PyTorch modified from [Shawn1993/cnn-text-classification-pytorch](https://github.com/Shawn1993/cnn-text-classification-pytorch).

Zhang's original implementation in Torch:
[https://github.com/zhangxiangxiao/Crepe](https://github.com/zhangxiangxiao/Crepe)

## Requirement
* python 2
* pytorch > 0.2
* numpy
* termcolor

## Train
```
python train.py -h
```

You will get:

```
Character-level CNN text classifier

optional arguments:
  -h, --help            show this help message and exit
  --train-path DIR      path to training data csv
  --val-path DIR        path to validating data csv

Learning options:
  --lr LR               initial learning rate [default: 0.0005]
  --epochs EPOCHS       number of epochs for train [default: 200]
  --batch-size BATCH_SIZE
                        batch size for training [default: 128]
  --max-norm MAX_NORM   Norm cutoff to prevent explosion of gradients
  --optimizer OPTIMIZER
                        Type of optimizer. SGD|Adam|ASGD are supported
                        [default: Adam]
  --class-weight        Weights should be a 1D Tensor assigning weight to each
                        of the classes.
  --dynamic-lr          Use dynamic learning schedule.
  --milestones MILESTONES [MILESTONES ...]
                        List of epoch indices. Must be increasing.
                        Default:[5,10,15]
  --decay-factor DECAY_FACTOR
                        Decay factor for reducing learning rate [default: 0.5]

Model options:
  --alphabet-path ALPHABET_PATH
                        Contains all characters for prediction
  --l0 L0               maximum length of input sequence to CNNs [default:
                        1014]
  --shuffle             shuffle the data every epoch
  --dropout DROPOUT     the probability for dropout [default: 0.5]
  -kernel-num KERNEL_NUM
                        number of each kind of kernel
  -kernel-sizes KERNEL_SIZES
                        comma-separated kernel size to use for convolution

Device options:
  --num-workers NUM_WORKERS
                        Number of workers used in data-loading
  --cuda                enable the gpu

Experiment options:
  --verbose             Turn on progress tracking per iteration for debugging
  --continue-from CONTINUE_FROM
                        Continue from checkpoint model
  --checkpoint          Enables checkpoint saving of model
  --checkpoint-per-batch CHECKPOINT_PER_BATCH
                        Save checkpoint per batch. 0 means never save
                        [default: 10000]
  --save-folder SAVE_FOLDER
                        Location to save epoch models, training configurations
                        and results.
  --log-config          Store experiment configuration
  --log-result          Store experiment result
  --log-interval LOG_INTERVAL
                        how many steps to wait before logging training status
                        [default: 1]
  --val-interval VAL_INTERVAL
                        how many steps to wait before vaidation [default: 200]
  --save-interval SAVE_INTERVAL
                        how many epochs to wait before saving [default:1]
```


```
python train.py
```
You will get:

```
Epoch[8] Batch[200] - loss: 0.237892  lr: 0.00050  acc: 93.7500%(120/128))
Evaluation - loss: 0.363364  acc: 89.1155%(6730/7552)
Label:   0      Prec:  93.2% (1636/1755)  Recall:  86.6% (1636/1890)  F-Score:  89.8%
Label:   1      Prec:  94.6% (1802/1905)  Recall:  95.6% (1802/1884)  F-Score:  95.1%
Label:   2      Prec:  85.6% (1587/1854)  Recall:  84.1% (1587/1888)  F-Score:  84.8%
Label:   3      Prec:  83.7% (1705/2038)  Recall:  90.2% (1705/1890)  F-Score:  86.8%
```

## Test
If you has construct you test set, you make testing like:

```
python test.py --test-path='data/ag_news_csv/test.csv' --model-path='models_CharCNN/CharCNN_best.pth.tar'
```
The model-path option means where your model load from.


## Reference
* Xiang Zhang, Junbo Zhao, Yann LeCun. [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). Advances in Neural Information Processing Systems 28 (NIPS 2015)

