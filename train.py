import torch.nn as nn
from model import Net
import numpy as np
import torch
from torchvision.datasets import mnist
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoch=100
batch_size = 256
learning_rate = 0.001
SAVE_MODEL_NAME='./model.pth'
def train():
    model = Net()
    model.to(device)
    model.train()
    train_dataset = mnist.MNIST(root='./data',train=True,transform=ToTensor(),download=True)
    test_dataset = mnist.MNIST(root='./data',train=False,transform=ToTensor(),download=True)
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

    optimizer=Adam(model.parameters(),lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    prev_acc=0

    for i in range(epoch):
        print("epoch:",i)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            if(batch_idx%50==0):
                print('batch_id:',batch_idx,loss.item())

        model.eval()
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(test_dataset)
        if np.abs(acc - prev_acc) < 2e-3:
             break
        prev_acc = acc


        print("accuracy:",acc)
        torch.save(model.state_dict(),SAVE_MODEL_NAME)

train()

#mnist图片的target本身就是一维张量也就是类别索引，和pred输出形状一致（猫狗模型训练中因为target是独热张量所以需要
#target.view_as(pred)把target转化成和pred一样的形状








