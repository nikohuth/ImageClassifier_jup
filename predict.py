# Imports here
import numpy as np
import torch
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from PIL import Image
from collections import OrderedDict
from torch.autograd import Variable
import signal
import argparse
from contextlib import contextmanager
import requests
import json
import herewegoagain

aperat = argparse.ArgumentParser(
    description='predict-file')
aperat.add_argument('input_img', default='flowers/test/42/image_05696.jpg', nargs=1, action="store", type = str)
aperat.add_argument('checkpoint', default='/home/workspace/ImageClassifier/checkpoint.pth', nargs='*', action="store",type = str)
aperat.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
aperat.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
aperat.add_argument('--gpu', default="gpu", action="store", dest="gpu",nargs='*')

pa = aperat.parse_args()
img_path = pa.input_img
outputs = pa.top_k
po = pa.gpu
input_img = pa.input_img
path = pa.checkpoint


model,_ = herewegoagain.load_checkpoint(path)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

probs, labels = herewegoagain.predict(img_path, model,outputs,po)

listerei = []
for i in labels:
    listerei += [cat_to_name[i]]
i = 0
while i < len(listerei):
    print("{} with prob of {}".format(listerei[i],probs[i]))
    i += 1