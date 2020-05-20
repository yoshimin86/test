import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm


fashion_train = FashionMNIST('./Fashion',train=True,download=True,transform=transforms.ToTensor())
fashion_test = FashionMNIST('./Fashion',train=False,download=True,transform=transforms.ToTensor())

batch_size = 128

train_loader = DataLoader(fashion_train,batch_size,shuffle=True)
test_loader = DataLoader(fashion_test,batch_size,shuffle=True)



#ネットワークの設定
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #畳み込み層
        chan1 = 24
        chan2 = 64
        fil = 5
        bec =int((((28-fil+1)/2)-fil+1)/2 * (((28-fil+1)/2)-fil+1)/2 * chan2)
        self.conv1 = nn.Conv2d(1,chan1,fil,padding=1)
        self.conv2 = nn.Conv2d(chan1,chan2,fil)
        self.bn1 = nn.BatchNorm2d(chan1)
        self.dropout1 = torch.nn.Dropout2d(p=0.5)
        self.dropout2 = torch.nn.Dropout(p=0.5)
        #全結合層
        self.fc1 = nn.Linear(bec,256)
        self.fc2 = nn.Linear(256,10)
       
    def forward(self,x,flag):
        chan2 = 64
        fil = 5
        bec =int((((28-fil+1)/2)-fil+1)/2 * (((28-fil+1)/2)-fil+1)/2 * chan2)
        #プーリング層
        if flag == 0:
            x = F.max_pool2d(F.sigmoid(self.bn1(self.conv1(x))),2)
        else:
            x = F.max_pool2d(F.sigmoid(self.conv1(x)),2)
        x = F.max_pool2d(F.sigmoid(self.conv2(x)),2)
        if flag==0:
            x = self.dropout1(x)
        x = x.view(-1,bec) #全結合層に入れるために256のベクトルに変換する
        x = F.sigmoid(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)



using_cuda = torch.cuda.is_available()
net = Net()

criterion = nn.NLLLoss()
if using_cuda:
    net.cuda()
    criterion.cuda()


loss_train_list = []
loss_test_list = []
acc_train =[]
acc_test = []
epochs = 50
for i in range(epochs):
    n_true_train = 0
    n_loss_train = 0
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    if i >25:
        optimizer = optim.Adam(net.parameters(), lr=0.0005)
    if i > 38:
        optimizer = optim.Adam(net.parameters(), lr=0.00001)
    for inputs, labels in train_loader:
        if using_cuda:
            x = inputs.cuda()
            y = labels.cuda()
            output = net(x,flag=0)
        else:
            x = inputs
            y = labels
            output = net(x,flag=0)
        predicted_train = torch.max(output.data, 1)
        if using_cuda:
            y_predicted_train = predicted_train.to('cpu').numpy()
        else:
            y_predicted_train = predicted_train.numpy()
        n_true_train += np.sum(y_predicted_train == y.to('cpu').numpy())
        optimizer.zero_grad()
        loss1 = criterion(output, y)
        n_loss_train += float(loss1)
        loss1.backward()
        optimizer.step()
    total_train = len(fashion_train)
    loss_train = n_loss_train / total_train
    accuracy_train = n_true_train / total_train
    print('epoch: {0}, loss_train : {1}, accuracy_train: {2}'.format(i, (float)(loss_train),(float)(accuracy_train)))
    loss_train_list.append(loss_train)
    acc_train.append(accuracy_train)

    # test
    n_loss_test = 0
    n_true_test = 0
    for inputs, labels in test_loader:
        if using_cuda:
            x = inputs.cuda()
            y = labels.cuda()
            output = net(x,flag=1)
        else:
            x = inputs
            y = labels
            output = net(x,flag=1)
        predicted = torch.max(output.data, 1)
        if using_cuda:
            y_predicted = predicted.to('cpu').numpy()
        else:
            y_predicted = predicted.numpy()
        n_true_test += np.sum(y_predicted == y.to('cpu').numpy())
        loss2 = criterion(output, y)
        n_loss_test += float(loss2)

    total_test = len(fashion_test)
    loss_test = n_loss_test / total_test
    accuracy_test = n_true_test / total_test
    print('epoch: {0}, loss_test : {1}, accuracy_test: {2}'.format(i, (float)(loss_test),(float)(accuracy_test)))
    loss_test_list.append(loss_test)
    acc_test.append(accuracy_test)


fig = plt.figure(figsize=(9,4))
plt.subplots_adjust(wspace=0.4)
x = np.arange(50)
ax = fig.add_subplot(1,2,1)
ax.set_title("Loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_ylim([0.0,0.03])
ax.plot(x,loss_train_list,color="blue",label="train")
ax.plot(x,loss_test_list,color="orange",label="test")
ax.legend()

ax = fig.add_subplot(1,2,2)
ax.set_title("Accuracy")
ax.set_xlabel("epoch")
ax.set_ylabel("accuracy")
ax.set_ylim([0.0,1.1])
ax.plot(x,acc_train,color="blue",label="train")
ax.plot(x,acc_test,color="orange",label="test")
ax.legend()

plt.savefig('fmnist20.jpg')
plt.show()
