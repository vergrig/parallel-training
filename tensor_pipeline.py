%%writefile tpipe.py
#!/usr/bin/env python

import os
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.multiprocessing import Process
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time
from itertools import chain
import numpy as np


def string_time(elapsed):
    return "%im %is" %(int(elapsed / 60), int(elapsed % 60))

def run_stage(rank, size):
    BATCH_SZ = 64
    FULL_SZ = 3 * 32 * 32
    MID_SZ = 40
    NUM_EPOCH = 1
    LOG_FREQ = 64
    running_loss, ind = 0, 0

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                    BATCH_SZ, shuffle=False, drop_last=True) 
    
    fc1 = nn.Linear(FULL_SZ // size, MID_SZ)
    relu = nn.ReLU()
    fc2 = nn.Linear(MID_SZ // size, 10)

    optimizer = torch.optim.Adam(chain(fc1.parameters(), 
                                       fc2.parameters()), lr=0.001)

    x = torch.zeros((BATCH_SZ, FULL_SZ // size))
    x2 = torch.zeros((BATCH_SZ, MID_SZ // size))

    for epoch in range(NUM_EPOCH):
        for batch in trainloader:
            inputs, labels = batch
            inputs = inputs.view(BATCH_SZ, -1)
            
            x = list(torch.split(inputs, (inputs.shape[1]
                                          + size - 1) // size, dim=1))[rank]
            mid = fc1(x)
            dist.all_reduce(mid)
            mid = relu(mid)

            x2 = list(torch.split(mid, (mid.shape[1] 
                                        + size - 1) // size, dim=1))[rank]
            output = fc2(x2)
            dist.all_reduce(output)

            loss = nn.CrossEntropyLoss()(output, labels)

            if rank == 0:
                running_loss += loss.item()
                ind += 1
                if ind % LOG_FREQ == 0:
                    print("Loss:", running_loss / LOG_FREQ)
                    running_loss = 0

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()


def init_process(local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_stage, backend='gloo')