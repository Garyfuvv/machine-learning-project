import torch
import random
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.io as io
import os
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter() #
 
if not os.path.exists('./img'):
    os.mkdir('./img')
 
class MyData(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transform, target_transform=None):
        super(MyData, self).__init__()
        file_txt = open(datatxt,'r')
        imgs = []
        for line in file_txt:
            line = line.rstrip()
            words = line.split('|')
            imgs.append((words[0], words[1]))
 
        self.imgs = imgs
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
 
    def __getitem__(self, index):
        #random.shuffle(self.imgs)
        name, label = self.imgs[index]
        img = Image.open(self.root + name).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = int(label)
        #label_tensor = torch.Tensor([0,0])
        #label_tensor[label]=1
        return img, label
 
    def __len__(self):
        return len(self.imgs)
 
def saveimg(imgs,name,n):
    # img = img / 2 + 0.5     # unnormalize
    for j in range(n):
        img = imgs[0,j,:,:]
        npimg = img.numpy()
        plt.imsave(name+'_%d.jpg'%j, npimg)
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3),  # input_size=(3*256*256)，padding=2
            nn.ReLU(),  # input_size=(32*256*256)
            nn.MaxPool2d((2, 2)))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)) )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)))
        self.fc1= nn.Sequential(
            nn.Linear(29 * 29 * 32, 120),
            nn.ReLU(), )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 60),
            nn.ReLU(),  )
        self.fc3 = nn.Sequential(
            nn.Linear(60, 2),  )
    def forward(self, x):
        x = self.conv1(x)
        print(np.shape(x))
        saveimg(x,'./img/conv1',4)
        x = self.conv2(x)
        print(np.shape(x))
        saveimg(x,'./img/conv2',16)
        x = self.conv3(x)
        print(np.shape(x))
        saveimg(x,'./img/conv3',32)
        x = x.view(-1, 29*29*32)
        x = self.fc1(x)
        # saveimg(x,'./img/fc1.jpg')
        x = self.fc2(x)
        # saveimg(x,'./img/fc2.jpg')
        x = self.fc3(x)
        # saveimg(x,'./img/fc3.jpg')
        return x
 
classes = ['NO','OK']
 
#ToTensor()能够把灰度范围从0-255变换到0-1之间，而后面的transform.Normalize()则把0-1变换到(-1,1).
transform = transforms.Compose(
    [transforms.Resize((250, 250)),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data=MyData(root ='train/',datatxt='train.txt', transform=transform)
 
 
test_data=MyData(root ='test/',datatxt='test.txt',transform=transform)
 
val_data=MyData(root ='val/',datatxt='val.txt',transform=transform)
 
 
 
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=20, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1,shuffle=False)
val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=20,shuffle=True)
 
def imshow(img,i):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imsave('%d.jpg'%i, np.transpose(npimg, (1, 2, 0)))
    # plt.show()
 
dataiter = iter(train_loader)
images, labels = dataiter.next()
# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch  Training')
parser.add_argument('--model_dir', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
args = parser.parse_args()
net = Net()
outf = "./model/"
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

phase = 'val'

if __name__ == "__main__":
 
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            if phase == 'train':
                for epoch in range(40):
                    print('\nEpoch: %d' % (epoch + 1))
                    net.train()
                    sum_loss = 0.0
                    correct = 0.0
                    total = 0.0
                    correct_epoch = 0.0
                    for i, data in enumerate(train_loader, 0):
                        # 准备数据
                        length = len(train_loader)
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        optimizer.zero_grad()

                        # forward + backward
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()

                        # 每训练1个batch打印一次loss和准确率
                        sum_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += predicted.eq(labels.data).cpu().sum()
                        print('[epoch:%d, iter:%d] Loss: %.03f '
                              % (epoch + 1, (i + 1 ), sum_loss / (i + 1)))
                        #f2.write('%03d  %05d |Loss: %.03f '% (epoch + 1, (i + 1 ), sum_loss / (i + 1)))
                        f2.write('Loss: %.03f ' % ( sum_loss / (i + 1)))
                        f2.write('\n')
                        f2.flush()
                    print('训练分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    #f.write("EPOCH=%03d,Train_Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write("Train_Accuracy= %.3f" % (acc))
                    f.write('\n')
                    f.flush()
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.model_dir, epoch + 1))

                    writer.add_scalar('loss', sum_loss / (i + 1), epoch+1) # add the information to the log file
                    writer.add_scalar('acc', acc, epoch+1)
                print("Training Finished, TotalEPOCH=%d" % epoch)

                # #d打印网络结构及参数和输出形状
                # net = net.to(device)
                # summary(net, input_size=(3, 250, 250))   #summary(net,(3,250,250))
            elif phase == 'val':
                #训练之后将所有的模型进行测试
                for i in range(25):

                    #print("测试val")
                    dataiter = iter(val_loader)
                    images, labels = dataiter.next()
                    #PATH =  'net' + '{}.pth'.format(39)
                    net = Net()
                    model = torch.load(os.path.join(args.model_dir, 'net_%03d.pth' % ( i + 1)))
                    #model = torch.load(os.path.join('net_%03d.pth' % 39))
                    net.load_state_dict(model)

                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for data in val_loader:
                            net.eval()
                            images, labels = data
                            #imshow(torchvision.utils.make_grid(images))
                            # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(6)))
                            # print(images)

                            #print("真实标签", labels)
                            outputs = net(images)



                            # torch.max(a, 1)返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
                            _, predicted = torch.max(outputs.data, 1)
                            #print("预测标签", predicted)
                            #print(outputs)

                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                    #print('Accuracy of the network on the 10 test images: %d %%' % (100 * correct / total))
                    print('%d' % (100 * correct / total))

                # model = torch.load(os.path.join(args.model_dir, 'net_%03d.pth' % ( 40)))
                # model = torch.load(os.path.join('net_%03d.pth' % 39))
                # net.load_state_dict(model)

            elif phase == 'test':
                print("测试test")

                dataiter = iter(test_loader)
                images, labels = dataiter.next()



                net = Net()
                model = torch.load(os.path.join(args.model_dir, 'net_%03d.pth' % (23)))
                net.load_state_dict(model)

                correct = 0
                total = 0
                with torch.no_grad():
                    for data in test_loader:
                        net.eval()
                        images, labels = data
                        imshow(torchvision.utils.make_grid(images),total)
                        # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(6)))
                        # print(images)

                        print("真实标签", labels)
                        outputs = net(images)



                        # torch.max(a, 1)返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
                        _, predicted = torch.max(outputs.data, 1)
                        print("预测标签", predicted)
                        #print(outputs)

                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the 40 test images: %d %%' % (100 * correct / total))
    writer.close()
