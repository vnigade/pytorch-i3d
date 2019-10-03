import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-load_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('-gpu', type=str)
parser.add_argument('-save_dir', type=str)
parser.add_argument('-split', type=str)
parser.add_argument('-overwrite', default=0)

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
from pytorch_i3d import InceptionI3d

from charades_dataset_full import Charades as Dataset


_WINDOW_SIZE = 64
_WINDOW_STRIDE = 16

def run(max_steps=64e3, mode='rgb', root='/ssd2/charades/Charades_v1_rgb', split='charades/charades.json', batch_size=1, load_model='', save_dir=''):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = Dataset(split, 'testing', root, mode, test_transforms, num=-1, save_dir=save_dir)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(load_model))
    print("Loaded model from checkpoint {0}".format(load_model))
    i3d.cuda()
    # i3d.cpu()

    for phase in ['train', 'val']:
        i3d.train(False)  # Set model to evaluate mode
                
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
                    
        # Iterate over data.
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
                with torch.no_grad():
                  ip = Variable(torch.from_numpy(inputs[:,:,window_start:window_end]).cuda())
                  # features = i3d(ip)
                  features = i3d.forward(ip)
                  # We need to reduce the results across time frames.
                  features = torch.mean(features, dim=2)
                output_dict["window_" + str(idx)]["scores"] = features.cpu().detach().numpy().flatten().tolist()
                idx += 1
                window_start += _WINDOW_STRIDE
                window_end = window_start + _WINDOW_SIZE
            with open(save_dir + "/" + name[0], 'w') as outfile:
                json.dump(output_dict, outfile)
                print("{0} Scores saved for {1}".format(len(output_dict), name[0]))

if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir, split=args.split)
