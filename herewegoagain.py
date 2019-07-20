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

arch = {"vgg19": 25088,
        "vgg13": 25088,
        "alexnet": 9216}



def load(wogehts = "flowers"):
    
    data_dirs = wogehts
    datat = ''
    for i in range(len(data_dirs)):
        datat += str(data_dirs[i])
    train_dir = datat + '/train'
    valid_dir = datat + '/valid'
    test_dir = datat + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([
                                        transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                  ])

    validation_transform = transforms.Compose([transforms.RandomRotation(256),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                  ])
    testing_transform = transforms.Compose([transforms.RandomRotation(256),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                  ])

    # TODO: Load the datasets with ImageFolder
    training_datasets = datasets.ImageFolder(train_dir, transform = training_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir, transform = validation_transform)
    testing_datasets = datasets.ImageFolder(test_dir, transform = testing_transform)
   


    # TODO: Using the image datasets and the trainforms, define the dataloaders

    trainloader = torch.utils.data.DataLoader(training_datasets, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_datasets, batch_size=32)
    testingloader = torch.utils.data.DataLoader(testing_datasets, batch_size=32)
    return trainloader, validationloader, testingloader, training_datasets

def setup(structure = 'vgg19', dropout = 0.5, hidden_layer = 120, learn = 0.001, po = 'gpu'):
    if structure == 'vgg19':
        model = models.vgg19(pretrained = True)
    elif structure == 'vgg13':
        model = models.vgg13(pretrained = True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print( "Sorry not available")

   
    for param in model.parameters():
        param.requires_grad = False
    #Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(arch[structure], hidden_layer)),
                            ('drop', nn.Dropout(dropout)),
                            ('hidden_layer1', nn.Linear(hidden_layer, 90)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(90, 102)),
                              ('output', nn.LogSoftmax(dim=1))    
    ]))

    model.classifier = classifier


    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),learn)
    for param in model.parameters():
        param.requires_grad = False


    if torch.cuda.is_available() and po == 'gpu':
        model.cuda()
    return model, criterion, optimizer


def train(model, criterion, optimizer, trainloader, v_loader, po = 'gpu', epoch = 1):
    epochs = epoch
    steps = 0
    running_loss = 0
    if  torch.cuda.is_available() and po == 'gpu':
        model.to('cuda')
    print_every = 20
    for param in model.parameters():
        param.requires_grad = True   
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
                # Move input and label tensors to the default device
                #inputs, labels = inputs.to(device), labels.to(device)
            model.train()
            if  torch.cuda.is_available() and po == 'gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in v_loader:
                        optimizer.zero_grad()
                        if  torch.cuda.is_available() and po == 'gpu':
                            inputs1, labels1 = inputs.to('cuda'), labels.to('cuda')
                        logps = model.forward(inputs1)
                        batch_loss = criterion(logps, labels1)
                        test_loss += batch_loss.item()
                                    # Calculate accuracy
                        ps = torch.exp(logps).data
                        equals = (labels1.data == ps.max(1)[1])
                        accuracy += equals.type_as(torch.cuda.FloatTensor()).mean()
                print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(v_loader):.3f}.. "
                          f"Test accuracy: {accuracy/len(v_loader):.3f}")
                running_loss = 0
                model.train()

def save(data, path = 'checkpoint', structure = 'vgg19', hidden_layer = 130, dropout= 0.5, learn= 0.01, epochs = 1):
    
    _,_,_,training_datasets = load(data)
    model,_,optimizer = setup(structure,dropout,hidden_layer,learn,'gpu')
    model.class_to_idx = training_datasets.class_to_idx
    checkpoint = {'classifier': model.classifier,
                  'hidden_layer': hidden_layer,
                  'structure': structure,
                    'input_size': arch[structure],
                  'output_size': 102,
                  'learning_rate': learn,
                  'dropout' : dropout,
                  'epochs' : epochs,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                'class_to_idx' : model.class_to_idx}
    torch.save(checkpoint, path)
    
    
def load_checkpoint(filepath):
    weg = ""
    for idx in filepath:
        weg += str(idx)
    if  torch.cuda.is_available():
        checkpoint = torch.load(weg)
    else:
        checkpoint = torch.load(weg, map_location = 'cpu')
 
    learning_rate = checkpoint['learning_rate']
    structure = checkpoint['structure']
    hidden_layer =  checkpoint['hidden_layer']
    dropout = checkpoint['dropout']
    
    
    model,_,optimizer = setup(structure, dropout, hidden_layer,learning_rate, 'gpu')
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    #model, optimizer = load_checkpoint('checkpoint.pth')
    return model,optimizer


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #check from lesson deep learning with pytorch
   
    photo = Image.open(image)
    photo = photo.resize((256,256))
    
    photo = photo.crop((16,16,240,240))
    photo = np.array(photo)/255
    
    
    means= np.array([0.485, 0.456, 0.406])
    standard_div = np.array([0.229, 0.224, 0.225])
    photo = (photo-means)/standard_div
    return photo.transpose(2,0,1)
    
    
    
def imshow(image, ax=None, title=None):
#check from lesson deep learning with pytorch
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1) 
    ax.imshow(image)
    return ax

def predict(image_path, model, topk=5, po = 'gpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implementa the code to predict the class from an image file
    if torch.cuda.is_available() and po == 'gpu':
        model.cuda()
    model.eval()
    weg = ""
    for idx in image_path:
        weg += str(idx)
  #put image in the right format (check Floattensor)
    image = process_image(weg)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    if torch.cuda.is_available() and po == 'gpu':
        image = image.cuda()
    image = image.unsqueeze_(0)
    output = model.forward(image)
    probabilitys = torch.exp(output).data
    probs = list(torch.topk(probabilitys,topk)[0])[0]
    idx = list(torch.topk(probabilitys,topk)[1])[0]
    platzhalter = [probs,idx]
    
    liste= []
    nextone = []
    nextee = list(model.class_to_idx.items())
    for i in range(len(nextee)):
        liste.append(nextee[i][0])
    
    for i in range(topk):
        nextone.append(liste[idx[i]])
    
    return platzhalter[0], nextone