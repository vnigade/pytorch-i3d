"""
Initialize the model module
New models can be defined by adding scripts under models/
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.models as tmodels
import importlib


def create_model(arch, nclass, pretrained, distributed):
    if arch in tmodels.__dict__:  # torchvision models
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = tmodels.__dict__[arch](pretrained=True)
            model = model.cuda()
        else:
            print("=> creating model '{}'".format(arch))
            model = tmodels.__dict__[arch]()
    else:  # defined as script in this directory
        model = importlib.import_module('.'+arch, package='models').model
        if not pretrained_weights == '':
            print('loading pretrained-weights from {}'.format(pretrained_weights))
            model.load_state_dict(torch.load(pretrained_weights))

    # replace last layer
    if hasattr(model, 'classifier'):
        newcls = list(model.classifier.children())
        newcls = newcls[:-1] + [nn.Linear(newcls[-1].in_features, nclass).cuda()]
        model.classifier = nn.Sequential(*newcls)
    elif hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, nclass)
        if hasattr(model, 'AuxLogits'):
            model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, nclass)
    else:
        newcls = list(model.children())
        if hasattr(model, 'in_features'):
            in_features = model.in_features
        else:
            in_features = newcls[-1].in_features
        newcls = newcls[:-1] + [nn.Linear(in_features, nclass).cuda()]
        model = nn.Sequential(*newcls)

    if distributed:
        dist.init_process_group(backend=dist_backend, init_method=dist_url,
                                world_size=world_size)
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        if hasattr(model, 'features'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-3,
                                momentum=0.9,
                                weight_decay=1e-4)
   
    return model, criterion, optimizer
