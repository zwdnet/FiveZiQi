# coding:utf-8
# 测试GPU加速
# 参考:https://blog.csdn.net/qq_35149632/article/details/105236545


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.9
n_epochs = 3
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


# 加载数据
def loaddata():
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist', train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize((0.1307,),(0.3081,))])), batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader( torchvision.datasets.MNIST('./mnist', train=False, download=True, transform=torchvision.transforms.Compose(
[torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,),(0.3081,))])), batch_size=batch_size_test, shuffle=True)
    # example = enumerate(train_loader)
    #bacth_idx, (example_data, example_targets) = next(example)
    #print(example_data.shape)
    return (train_loader, test_loader)


# 搭建网络
class Cnet(nn.Module):
    def __init__(self):
        super(Cnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*128, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.Linear(1000, 10)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 7*7*128)
        x = self.fc(x)
        return F.log_softmax(x)
        
        
# CPU版本
def CPU(train_data_loader, test_data_loader):
    print("CPU版")
    network = Cnet()
    optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum=momentum)
    loss_func = torch.nn.NLLLoss()
    train_loss_his = []
    test_loss_his = []
    
    def train(epoch):
        network.train()
        for i, (batch_x, batch_y) in enumerate(train_data_loader):
            output = network(batch_x)
            loss = loss_func(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print('Epoch|%d, loss|%f'%(i, loss.item()))
                train_loss_his.append(loss.item())
                
    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                output = network(batch_x)
                test_loss += F.nll_loss(output, batch_y, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(batch_y.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            print('loss|%.4f, pred|%.5f'%(test_loss, 100. * correct / len(test_loader.dataset)))
            
    # 实际测试代码
    starttime = time.time()
    time.sleep(2.1)
    for i in range(1, n_epochs+1):
        train(i)
        test()
    endtime = time.time()
    dtime = endtime - starttime
    
    
# GPU版本
def CPU(train_data_loader, test_data_loader):
    print("GPU版")
    network = Cnet()
    network = network.cuda()
    optimizer = optim.SGD(network.parameters(), lr = learning_rate, momentum=momentum)
    loss_func = torch.nn.NLLLoss()
    loss_func = loss_func.cuda()
    train_loss_his = []
    test_loss_his = []
    
    def train(epoch):
        network.train()
        for i, (batch_x, batch_y) in enumerate(train_data_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            output = network(batch_x)
            loss = loss_func(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.cpu()
            
            if i % 100 == 0:
                print('Epoch|%d, loss|%f'%(i, loss.item()))
                train_loss_his.append(loss.item())
                
    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                output = network(batch_x)
                            loss = loss_func(output, y)
            test_loss += loss.cpu().item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        print('loss|%.4f, pred|%.5f'%(test_loss, 100. * correct / len(test_loader.dataset)))
            
    # 实际测试代码
    starttime = time.time()
    time.sleep(2.1)
    for i in range(1, n_epochs+1):
        train(i)
        test()
    endtime = time.time()
    dtime = endtime - starttime


if __name__ == "__main__":
    train_loader, test_loader = loaddata()
    CPU(train_loader, test_loader)
    train_loader, test_loader = loaddata()
    GPU(train_loader, test_loader)
    