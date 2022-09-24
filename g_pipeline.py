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
import numpy as np

def string_time(elapsed):
    return "%im %is" %(int(elapsed / 60), int(elapsed % 60))

def run_stage(rank, size):
    torch.manual_seed(11)

    if rank == 0:
        net = nn.Sequential(nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2))
    elif rank == 1:
        net = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2))
    elif rank == 2:
        net = nn.Sequential(nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU())
    elif rank == 3:
        net = nn.Sequential(nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))

    #optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    FIRST, LAST = 0, size - 1
    PREV, NEXT = rank - 1, rank + 1

    TOTAL_BATCH_SZ = 64
    MINI_BATCH_SZ = 16
    MINI_BATCH_NUM = TOTAL_BATCH_SZ // MINI_BATCH_SZ
    LOG_FREQ = 64

    NUM_EPOCH = 5
    device = 'cpu'

    rank_shapes = [[], [MINI_BATCH_SZ, 6, 14, 14], 
                   [MINI_BATCH_SZ, 16, 5, 5], 
                   [MINI_BATCH_SZ, 120], [1]]
    inputs = torch.zeros(rank_shapes[rank]).to(device) 
    labels = torch.zeros((TOTAL_BATCH_SZ), dtype=torch.long).to(device)
    grad = torch.zeros(rank_shapes[rank + 1]).to(device)
    signal_work = torch.ones(1)

    # i separated the code for different kinds of workers (first, mid, last)
    # this makes the code larger, but greatly improves readability

    if rank == FIRST:
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                        download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
                        TOTAL_BATCH_SZ, shuffle=False, drop_last=True) 
        
        for epoch in range(NUM_EPOCH):
            for big_batch in trainloader:
                b_inputs = torch.split(big_batch[0], MINI_BATCH_SZ)
                #b_labels = torch.split(big_batch[1], MINI_BATCH_SZ)

                dist.send(tensor=signal_work, dst=NEXT)
                batch_inputs = []

                for i in range(MINI_BATCH_NUM):
                    inputs = b_inputs[i]

                    batch_inputs.append(deepcopy(inputs))
                    batch_inputs[-1].requires_grad = True
                    outputs = net(batch_inputs[-1])

                    dist.send(tensor=outputs, dst=NEXT)

                dist.send(tensor=big_batch[1], dst=LAST)

                for i in range(MINI_BATCH_NUM):
                    dist.recv(tensor=grad, src=NEXT)
                    net(batch_inputs[-1 - i]).backward(grad)

                dist.barrier()
                optimizer.step()
                optimizer.zero_grad()
                dist.barrier()

        signal_work = -signal_work
        dist.send(tensor=signal_work, dst=NEXT)

    elif rank != LAST:
        while True:
            dist.recv(tensor=signal_work, src=PREV)
            dist.send(tensor=signal_work, dst=NEXT)
            if signal_work.item() == -1:
                break

            batch_inputs = []

            for i in range(MINI_BATCH_NUM):
                dist.recv(tensor=inputs, src=PREV)
                batch_inputs.append(deepcopy(inputs.detach()))
                batch_inputs[-1].requires_grad = True
                outputs = net(batch_inputs[-1])

                dist.send(tensor=outputs, dst=NEXT)
                
            for i in range(MINI_BATCH_NUM):
                dist.recv(tensor=grad, src=NEXT)
                net(batch_inputs[-1 - i]).backward(grad)
                grad_new = batch_inputs[-1 - i].grad

                dist.send(tensor=grad_new, dst=PREV)

            dist.barrier()
            optimizer.step()
            optimizer.zero_grad()
            dist.barrier()
    
    else:
        start = time()
        running_loss = []
        logging_loss, b_ind = [0], 0
        criterion = nn.CrossEntropyLoss()

        while True:
            dist.recv(tensor=signal_work, src=PREV)
            if signal_work.item() == -1:
                break

            batch_inputs = []

            for i in range(MINI_BATCH_NUM):
                dist.recv(tensor=inputs, src=PREV)

                batch_inputs.append(deepcopy(inputs.detach()))
                batch_inputs[-1].requires_grad = True

            dist.recv(tensor=labels, src=FIRST)
            
            batch_labels = torch.split(labels.detach(), MINI_BATCH_SZ)

            local_loss = []
            for i in range(MINI_BATCH_NUM):
                loss = criterion(net(batch_inputs[-1 - i]), batch_labels[-1 - i])
                local_loss.append(loss.item())

                loss.backward()
                grad_new = batch_inputs[-1 - i].grad

                dist.send(tensor=grad_new, dst=PREV)
            
            b_ind += 1
            logging_loss[-1] += sum(local_loss) / MINI_BATCH_NUM
            if b_ind % LOG_FREQ == LOG_FREQ - 1:
                logging_loss[-1] /= LOG_FREQ
                print("Loss:", logging_loss[-1], "Elapsed time: " + string_time(time() - start))
                logging_loss.append(0)
            
            running_loss.append(sum(local_loss) / MINI_BATCH_NUM)

            dist.barrier()
            optimizer.step()
            optimizer.zero_grad()
            dist.barrier()

        np.save("run_log.npy", np.array(running_loss))
        np.save("progress.npy", np.array(logging_loss))


def init_process(local_rank, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    dist.init_process_group(backend, rank=local_rank)
    size = dist.get_world_size()
    fn(local_rank, size)

if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    init_process(local_rank, fn=run_stage, backend='gloo')