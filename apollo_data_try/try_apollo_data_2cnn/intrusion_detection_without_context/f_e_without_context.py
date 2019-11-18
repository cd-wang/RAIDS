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
from data_extraction_ABS import *
from time import gmtime, strftime

face_dataset = intrusion_data(csv_file='interpolated_0to24561_random_attack_a_0p08_0p5_17192.csv', root_dir='./data', cs=1,
                              transform=transforms.Compose(
                                  [transforms.Resize(256),
                                   transforms.RandomResizedCrop(224),
                                   transforms.ToTensor()]))

test_dataset = intrusion_data(csv_file='interpolated_0to24561_random_attack_a_0p08_0p5_7370.csv', root_dir='./data', cs=1,
                              transform=transforms.Compose(
                                  [transforms.Resize(256),
                                   transforms.RandomResizedCrop(224),
                                   transforms.ToTensor()]))
# out=[]
dataloader = DataLoader(face_dataset, batch_size=4, shuffle=False)
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

epochs = 100
cnt = 0
for iter_x in range(epochs):
    inter = .10
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
    for i_batch, sample in enumerate(dataloader):  # for each training i_batch
        if i_batch/len(dataloader)> inter:
            print('completed %epoch ',inter)
            inter+=.10

        data=[]
        result=[]
        for sm in sample:
            imag , dat , res = sm

            data.append(dat.type(torch.FloatTensor))
            result.append(res)
        net.zero_grad()
        target=torch.stack(result)
        target=target.view(-1)

        final_vars=[]
        for tens in data:
            final_vars.append(Variable(tens))
        x=net(*final_vars)
        values, indices = x.max(1)
        loss=criterion(x,Variable(target))
        # train_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        if(i_batch>10):
            break

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&TESTING &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    print("test_start: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    for i_batch, sample in enumerate(test_loader):
        data=[]
        result=[]
        for sm in sample:
            imag , dat , res = sm
            data.append(dat.type(torch.FloatTensor))
            result.append(res)
        target=torch.stack(result)
        target=target.view(-1)
        final_vars=[]
        for tens in data:
            final_vars.append(Variable(tens))

        x = net(*final_vars)

        values, indices = x.max(1)
        print("target_indices: ", target, indices.data)
        loss1 += accuracy(target,indices.data)
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

        # if(i_batch>10):
        #     break
    
    acc = 1.0-(loss1/finale)
    acc11 = target_1_indices_1/(target_1_indices_1+target_1_indices_0)
    print('accuracy=',acc)
    print('accuracy11=',acc11)
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

    # net.save_state_dict('mytraining.pt')
    filenm = 'epoch_' + str(cnt) + '.pt'
    # print(filenm)
    torch.save(net, os.path.join(('saved_models_random'), filenm))

    cnt=cnt+1
    if(cnt%10==0):
        print('EPOCH completed by %',(cnt/40)*100)

    print("test_end: ", strftime("%Y-%m-%d %H:%M:%S", gmtime()))



