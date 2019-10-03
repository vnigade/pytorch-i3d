import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-arch', type=str, help='vgg16')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)
parser.add_argument('-split', type=str)
parser.add_argument('-overwrite', default=0)
parser.add_argument('-nclass', default=157)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

import numpy as np
from collections import defaultdict
import json

from charades_dataset_full import Charades as Dataset
from models import create_model
import checkpoints

_WINDOW_SIZE = 64
_WINDOW_STRIDE = 16
_GAP = _WINDOW_SIZE / 4


def run(max_steps=64e3, arch='vgg16', mode='rgb', root='/ssd2/charades/Charades_v1_rgb', split='charades/charades.json', batch_size=1, load_model='', save_dir=''):
    # setup dataset
    # test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    inputsize = 224
    test_transforms = transforms.Compose([transforms.Resize(int(256./224*inputsize)),
                                  transforms.CenterCrop(inputsize),
                                  transforms.ToTensor(),
                                  normalize])

    dataloaders = {}
    datasets = {}
    phases = ['train']
    if 'train' in phases:
      dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir, rescale=False, model='vgg16')
      dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
      dataloaders['train'] = dataloader
      datasets['train'] = dataset      

    if 'val' in phases:
      val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir, rescale=False, model='vgg16')
      val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
      dataloaders['val'] = val_dataloader
      datasets['val'] = val_dataset      

    # dataloaders = {'train': dataloader, 'val': val_dataloader}
    # datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    model, criterion, optimizer = create_model(arch, nclass=args.nclass, pretrained=True, distributed=False)
    checkpoints.load(load_model, model, optimizer)
    print("Loaded model from checkpoint {0}".format(load_model))
    print(model)

    # phases = ['train', 'val']
    test_transforms = transforms.Compose([transforms.ToTensor(), normalize])
    for phase in phases:
        model.train(False)  # Set model to evaluate mode
                
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
                    
        # Iterate over data.
        print("Phase is {}".format(phase))
        spacing = np.linspace(0, _WINDOW_SIZE, _GAP, dtype=int, endpoint=False)
        for data in dataloaders[phase]:
            # get the inputs
            inputs, labels, name = data
            if args.overwrite == 0 and os.path.exists(os.path.join(save_dir, name[0])):
                continue
            
            b,c,t,h,w = inputs.shape
            inputs = inputs.numpy()
            if t < _WINDOW_SIZE:
                # Append start frames to the input
                extra_frames = inputs[:,:,0:(_WINDOW_SIZE - t),:,:]
                inputs = np.concatenate((inputs, extra_frames), axis=2)

            # process each window
            window_start = 0
            window_end = window_start + _WINDOW_SIZE
            output_dict = defaultdict(lambda: defaultdict(list))
            idx = 0
            while window_end < t:
                indexes = spacing + window_start
                input = np.squeeze(inputs[:,:,indexes], axis=0)
                input = np.transpose(input, (1, 2, 3, 0))
                # transform is performed on single image at a time
                # input_batch = [] 
                # for i in range(input.shape[0]):
                #   input_batch.append(test_transforms(input[i])) #.unsqueeze_(0)
                # input_batch = torch.stack(input_batch)
                input_batch = input

                with torch.no_grad():
                  ip = Variable(torch.from_numpy(input_batch).cuda())
                # features = i3d(ip)
                  features = model(ip)
                  # Perform softmax and then average the score for the video/window level prediction
                  features = torch.nn.Softmax(dim=1)(features)  
                  features = torch.mean(features, dim=0)
                output_dict["window_" + str(idx)]["scores"] = features.cpu().detach().numpy().flatten().tolist()
                idx += 1
                window_start += _WINDOW_STRIDE
                window_end = window_start + _WINDOW_SIZE
            with open(save_dir + "/" + name[0], 'w') as outfile:
                json.dump(output_dict, outfile)
                print("{0} Scores saved for {1}".format(len(output_dict), name[0]))

if __name__ == '__main__':
    # need to add argparse
    run(arch=args.arch, root=args.root, load_model=args.load_model, save_dir=args.save_dir, split=args.split)
