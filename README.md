## Introduction
This is the implementation of Kim's [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) paper in PyTorch.

1. Kim's implementation of the model in Theano:
[https://github.com/yoonkim/CNN_sentence](https://github.com/yoonkim/CNN_sentence)
2. Denny Britz has an implementation in Tensorflow:
[https://github.com/dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)

## Requirement
* python 3
* pytorch > 0.1
* torchtext > 0.1
* numpy

## Result
I just tried two dataset, MR and SST.

|Dataset|Class Size|Best Result|Kim's Paper Result|
|---|---|---|---|
|MR|2|77.5%(CNN-rand-static)|76.1%(CNN-rand-nostatic)|
|SST|5|37.2%(CNN-rand-static)|45.0%(CNN-rand-nostatic)|

I haven't adjusted the hyper-parameters for SST seriously.

## Usage
```
./main.py -h
```
or 

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
/test.py -test -model-path="./models/2017-02-11_15-50-53/snapshot_steps1500.pt
```
The snapshot option means where your model load from. If you don't assign it, the model will start from scratch.

## Predict
* **Example1**

	```
	./main.py -predict="Hello my dear , I love you so much ." \
	          -snapshot="./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt" 
	```
	You will get:
	
	```
	Loading model from [./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt]...
	
	[Text]  Hello my dear , I love you so much .
	[Label] positive
	```
* **Example2**

	```
	./main.py -predict="You just make me so sad and I have to leave you ."\
	          -snapshot="./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt" 
	```
	You will get:
	
	```
	Loading model from [./snapshot/2017-02-11_15-50-53/snapshot_steps1500.pt]...
	
	[Text]  You just make me so sad and I have to leave you .
	[Label] negative
	```

Your text must be separated by space, even punctuation.And, your text should longer then the max kernel size.

## Reference
* Xiang Zhang, Junbo Zhao, Yann LeCun. [Character-level Convolutional Networks for Text Classification](http://arxiv.org/abs/1509.01626). Advances in Neural Information Processing Systems 28 (NIPS 2015)

