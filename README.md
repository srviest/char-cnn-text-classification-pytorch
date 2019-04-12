## Introduction
This is the implementation of Zhang's [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626) paper in PyTorch modified from [Shawn1993/cnn-text-classification-pytorch](https://github.com/Shawn1993/cnn-text-classification-pytorch).

Zhang's original implementation in Torch:
[https://github.com/zhangxiangxiao/Crepe](https://github.com/zhangxiangxiao/Crepe)

## Requirement
* python 2, 3
* pytorch >= 0.5
* numpy
* termcolor

## Dataset Format
Each sample looks like:  

```
"class idx","sentence or text to be classified"  
```  

Samples are separated by newline.  

Example:  

```
"3","Fears for T N pension after talks, Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
"4","The Race is On: Second Private Team Sets Launch Date for Human Spaceflight (SPACE.com)","SPACE.com - TORONTO, Canada -- A second\team of rocketeers competing for the  #36;10 million Ansari X Prize, a contest for\privately funded suborbital space flight, has officially announced the first\launch date for its manned rocket."
```

## Train
```
python train.py -h
```

You will get:

```
Character-level CNN text classifier

optional arguments:
  -h, --help            show this help message and exit
  --train_path DIR      path to training data csv
  --val_path DIR        path to validation data csv

Learning options:
  --lr LR               initial learning rate [default: 0.0001]
  --epochs EPOCHS       number of epochs for train [default: 200]
  --batch_size BATCH_SIZE
                        batch size for training [default: 64]
  --max_norm MAX_NORM   Norm cutoff to prevent explosion of gradients
  --optimizer OPTIMIZER
                        Type of optimizer. SGD|Adam|ASGD are supported
                        [default: Adam]
  --class_weight        Weights should be a 1D Tensor assigning weight to each
                        of the classes.
  --dynamic_lr          Use dynamic learning schedule.
  --milestones MILESTONES [MILESTONES ...]
                        List of epoch indices. Must be increasing.
                        Default:[5,10,15]
  --decay_factor DECAY_FACTOR
                        Decay factor for reducing learning rate [default: 0.5]

Model options:
  --alphabet_path ALPHABET_PATH
                        Contains all characters for prediction
  --l0 L0               maximum length of input sequence to CNNs [default:
                        1014]
  --shuffle             shuffle the data every epoch
  --dropout DROPOUT     the probability for dropout [default: 0.5]
  -kernel_num KERNEL_NUM
                        number of each kind of kernel
  -kernel_sizes KERNEL_SIZES
                        comma-separated kernel size to use for convolution

Device options:
  --num_workers NUM_WORKERS
                        Number of workers used in data-loading
  --cuda                enable the gpu

Experiment options:
  --verbose             Turn on progress tracking per iteration for debugging
  --continue_from CONTINUE_FROM
                        Continue from checkpoint model
  --checkpoint          Enables checkpoint saving of model
  --checkpoint_per_batch CHECKPOINT_PER_BATCH
                        Save checkpoint per batch. 0 means never save
                        [default: 10000]
  --save_folder SAVE_FOLDER
                        Location to save epoch models, training configurations
                        and results.
  --log_config          Store experiment configuration
  --log_result          Store experiment result
  --log_interval LOG_INTERVAL
                        how many steps to wait before logging training status
                        [default: 1]
  --val_interval VAL_INTERVAL
                        how many steps to wait before vaidation [default: 200]
  --save_interval SAVE_INTERVAL
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

