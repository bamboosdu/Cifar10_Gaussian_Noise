'''
Author: your name
Date: 2021-05-16 21:27:57
LastEditTime: 2021-05-16 22:41:34
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /cifar_generate/cifar10.py
'''



from PIL import Image
import numpy as np
import os
import pickle, cPickle
from random import randrange

COUNT=60000
DATA_LEN=3072
CHANNEL_LEN = 1024
SHAPE = 32
LABEL_NUM=0
mean=0.0
variance=1.0
BIN_COUNTS=6

def pickled(savepath, data, label, fnames, bin_num=BIN_COUNTS, mode="train"):
  '''
    savepath (str): save path
    data (array): image data, a nx3072 array
    label (list): image label, a list with length n
    fnames (str list): image names, a list with length n
    bin_num (int): save data in several files
    mode (str): {'train', 'test'}
  '''
  assert os.path.isdir(savepath)
  total_num = len(fnames)
  samples_per_bin = total_num / bin_num
  assert samples_per_bin > 0
  idx = 1
  for i in range(bin_num): 
    start = i*samples_per_bin
    end = (i+1)*samples_per_bin
    
    if end <= total_num:
      dict = {'data': data[start:end, :],
              'labels': label[start:end],
              'filenames': fnames[start:end]}
    else:
      dict = {'data': data[start:, :],
              'labels': label[start:],
              'filenames': fnames[start:]}
    if mode == "train":
      dict['batch_label'] = "training batch {} of {}".format(idx, bin_num)
    else:
      dict['batch_label'] = "testing batch {} of {}".format(idx, bin_num)
      
    with open(os.path.join(savepath, 'data_batch_'+str(idx)), 'wb') as fi:
      cPickle.dump(dict, fi)
    idx = idx + 1

def unpickled(filename):
  #assert os.path.isdir(filename)
  assert os.path.isfile(filename)
  with open(filename, 'rb') as fo:
    dict = cPickle.load(fo)
  return dict

def generate(mean,variance):
    data = np.zeros((COUNT, DATA_LEN), dtype=np.uint8)
    label=np.zeros(COUNT)
    lst=[str(i)+".jpg" for i in range(COUNT)]

    idx=0
    c=CHANNEL_LEN

    for i in range(COUNT):
    
        im=np.random.normal(mean, variance, (32,32,3))
    
        data[idx,:c] =  np.reshape(im[:,:,0], c)
        data[idx, c:2*c] = np.reshape(im[:,:,1], c)
        data[idx, 2*c:] = np.reshape(im[:,:,2], c)
    
        label[idx]=randrange(10)

        # input()
        idx=idx+1
    return data, label, lst


save_path='./bin'

data, label, lst=generate(0,1)
pickled(save_path, data, label, lst, bin_num = BIN_COUNTS, mode="train")

# print (unpickled('./bin/data_batch_0'))



