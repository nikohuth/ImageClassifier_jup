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
from contextlib import contextmanager

import requests
import argparse
import json 
import herewegoagain

aperat = argparse.ArgumentParser(description='Train.py')

aperat.add_argument('data_dir', nargs='*', action="store", default="flowers")
aperat.add_argument('--gpu',  nargs='*',dest="gpu", action="store", default="gpu")
aperat.add_argument('--save_dir', dest="save_dir", action="store", default="/home/workspace/ImageClassifier/checkpoint.pth")
aperat.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.01, type = float)
aperat.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
aperat.add_argument('--epochs', dest="epochs", action="store", type=int, default=2)
aperat.add_argument('--arch', dest="arch", action="store", default="vgg19", type = str)
aperat.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=140)

parat = aperat.parse_args()
wo = parat.data_dir
weg = parat.save_dir
lernen = parat.learning_rate
structure = parat.arch
dropout = parat.dropout
hidden_layer = parat.hidden_units
strom = parat.gpu
epochs = parat.epochs

trainloader, v_loader, testloader,_ = herewegoagain.load(wo)
model, optimizer, criterion = herewegoagain.setup(structure,dropout,hidden_layer,lernen,strom)
herewegoagain.train(model, optimizer, criterion, trainloader, v_loader, strom, epochs)
herewegoagain.save(wo,weg,structure,hidden_layer,dropout,lernen)