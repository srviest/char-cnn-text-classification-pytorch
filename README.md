## Introduction
This is the implementation of Zhang's [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626) paper in PyTorch.

Zhang's implementation of the model in Torch:
[https://github.com/zhangxiangxiao/Crepe](https://github.com/zhangxiangxiao/Crepe)

## Requirement
* python 2
* pytorch > 0.2
* numpy

## Basic Usage
```
python train.py -h
```

You will get:

```
Character-level CNN text classifier

optional arguments:
  -h, --help            show this help message and exit
  -lr LR                initial learning rate [default: 0.0005]
  -epochs EPOCHS        number of epochs for train [default: 200]
  -batch-size BATCH_SIZE
                        batch size for training [default: 128]
  -train-path DIR       path to training data csv
  -val-path DIR         path to validating data csv
  -alphabet-path ALPHABET_PATH
                        Contains all characters for prediction
  -shuffle              shuffle the data every epoch
  -dropout DROPOUT      the probability for dropout [default: 0.5]
  -max-norm MAX_NORM    l2 constraint of parameters [default: 3.0]
  -kernel-num KERNEL_NUM
                        number of each kind of kernel
  -kernel-sizes KERNEL_SIZES
                        comma-separated kernel size to use for convolution
  --num_workers NUM_WORKERS
                        Number of workers used in data-loading
  -device DEVICE        device to use for iterate data, -1 mean cpu [default:
                        -1]
  -cuda                 enable the gpu
  -verbose              Turn on progress tracking per iteration for debugging
  -checkpoint           Enables checkpoint saving of model
  -save-folder SAVE_FOLDER
                        Location to save epoch models
  -log-interval LOG_INTERVAL
                        how many steps to wait before logging training status
                        [default: 1]
  -test-interval TEST_INTERVAL
                        how many steps to wait before testing [default: 100]
  -save-interval SAVE_INTERVAL
                        how many epochs to wait before saving [default:20]
```

## Train
```
./train.py
```
You will get:

```
Epoch[1] Batch[300] - loss: 1.044269  lr: 0.00050  acc: 42.9688%(55/128)
Evaluation - loss: 0.016534  acc: 51.2712%(3872/7552)
```

## Test
If you has construct you test set, you make testing like:

```
/test.py -test -model-path="models/CharCNN_10.pth.tar"
```
The model-path option means where your model load from.

## Predict
* **Example1**

	```
	./predict.py -text="Hello my dear , I love you so much ." \
	          -model-path="models/CharCNN_10.pth.tar"
	```
	You will get:
	
	```
	Loading model from [models/CharCNN_10.pth.tar]...
	
	[Text]  Hello my dear , I love you so much .
	[Label] positive
	```
* **Example2**

	```
	./predicy.py -text="You just make me so sad and I have to leave you ."\
	          -model-path="models/CharCNN_10.pth.tar"
	```
	You will get:
	
	```
	Loading model from [./models/CharCNN_10.pth.tar']...
	
	[Text]  You just make me so sad and I have to leave you .
	[Label] negative
	```

Your text must be separated by space, even punctuation.And, your text should longer then the max kernel size.

## Reference
* Xiang Zhang, Junbo Zhao, Yann LeCun. [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). Advances in Neural Information Processing Systems 28 (NIPS 2015)

