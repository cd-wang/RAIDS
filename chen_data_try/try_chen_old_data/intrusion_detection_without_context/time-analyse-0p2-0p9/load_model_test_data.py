from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from data_extraction_ABS_60 import *
from model_abs_linear import *

from time import gmtime, strftime

print("intrusion_detection_start: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

test_dataset = intrusion_data(csv_file='interpolated_attack_a_directive_0p25_1_10702to24371_part1_4103.csv', root_dir='./data', cs=1,
                              transform=transforms.Compose(
                                  [transforms.Resize(256),
                                   transforms.RandomResizedCrop(224),
                                   transforms.ToTensor()]))

test_loader=DataLoader(test_dataset, batch_size=4, shuffle=False)
d_batch = 4

net=mynet()
# if use_gpu:
#     net.cuda()
criterion = F.nll_loss
optimizer = optim.Adam(net.parameters(), lr=0.01)


def accuracy(act , out):
    wrong1=0
    for a,f in zip(act,out):
        if a!=f:
            wrong1+=1
    return wrong1


net.eval()
net = torch.load('saved_models_random/epoch_22.pt')

loss1 = 0
finale = 0
train_loss = 0
finale_1 = 0
finale_0 = 0
get_1 = 0
get_0 = 0
target_1_indices_1 = 0
target_1_indices_0 = 0
target_0_indices_1 = 0
target_0_indices_0 = 0

for i_batch, sample in enumerate(test_loader):
    data = []
    result = []
    for sm in sample:
        imag, dat, res = sm
        data.append(dat.type(torch.FloatTensor))
        result.append(res)
    target = torch.stack(result)
    target = target.view(-1)
    final_vars = []
    for tens in data:
        final_vars.append(Variable(tens))

    x = net(*final_vars)

    values, indices = x.max(1)
    print("target_indices: ", target, indices.data)
    loss1 += accuracy(target, indices.data)
    finale += 4
    for Target, Indices in zip(target, indices.data):
        if Target == 1:
            finale_1 += 1
        else:
            finale_0 += 1
        if Indices == 1:
            get_1 += 1
        else:
            get_0 += 1
        if Target == 1 and Indices == 1:
            target_1_indices_1 += 1
        elif Target == 1 and Indices == 0:
            target_1_indices_0 += 1
        elif Target == 0 and Indices == 0:
            target_0_indices_0 += 1
        elif Target == 0 and Indices == 1:
            target_0_indices_1 += 1

acc = 1.0 - (loss1 / finale)
acc11 = target_1_indices_1 / (target_1_indices_1 + target_1_indices_0)
print('accuracy=', acc)
print('accuracy11=', acc11)
print("finale_1_0_get_1_0: ", finale_1, finale_0, get_1, get_0)
print("target_indices11,10,00,01: ", target_1_indices_1, target_1_indices_0, target_0_indices_0, target_0_indices_1)

file2write = open("accuracy_file_f_e_without_context_abs_linear.txt", 'a')
file2write.write("accuracy= " + str(acc) + "\n")
file2write.write("accuracy1111= " + str(acc11) + "\n")

file2write.write(
        "finale_1_0_get_1_0= " + "\n" + str(finale_1) + "\n" + str(finale_0) + "\n" + str(get_1) + "\n" + str(
            get_0) + "\n")
file2write.write(
        "target_indices11,10,00,01= " + "\n" + str(target_1_indices_1) + "\n" + str(target_1_indices_0) + "\n" + str(
            target_0_indices_0) + "\n" + str(target_0_indices_1) + "\n")
file2write.close()

print("intrusion_detection_end: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))

