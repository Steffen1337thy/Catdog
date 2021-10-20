import torch
import torchvision
from torchvision import transforms
from PIL import Image
from os import listdir
import random
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import os

train_path= 'data/train/'

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

cuda_count = torch.cuda.device_count()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    normalize])

class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3,6,kernel_size=5)
        self.conv2 = nn.Conv2d(6,12,kernel_size=5)
        self.conv3 = nn.Conv2d(12, 18, kernel_size=5)
        self.conv4 = nn.Conv2d(18, 24, kernel_size=5 )
        self.fc1 = nn.Linear(3456, 1000)
        self.fc2 = nn.Linear(1000,2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = x.view(-1, 3456)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

if os.path.isfile('Catsanddogs_model.pth'):
    model = torch.load('Catsanddogs_model.pth')
    model.eval()

    trained_model = True
else:
    trained_model = False

    #TARGET : [isCat,isDog]
    train_data_list = []
    target_list = []
    train_data = []
    files = listdir(train_path)
    for i in range(len(listdir(train_path))):
        f = random.choice(files)
        files.remove(f)
        img = Image.open(train_path + f)
        img_tensor = transform(img)
        train_data_list.append(img_tensor)
        isCat = 1 if 'cat' in f else 0
        isDog = 1 if 'dog' in f else 0
        target = [isCat,isDog]
        target_list.append(target)
        if len(train_data_list) >= 64:
            train_data.append((torch.stack(train_data_list), target_list))
            train_data_list = []
            target_list = []
            print('Loaded batch ', len(train_data), 'of ', int(1/64*len(listdir(train_path))))
            print('Percentage Done: ', len(train_data)*64 / int(len(listdir(train_path)))*100, '%' )
            # if len(train_data) > 2:
            #     break




model = Netz()
model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    batch_id = 0
    for data, target in train_data:
        data = data.cuda()
        target = torch.Tensor(target).cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.binary_cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id * len(data), len(listdir(train_path)),
            100. * batch_id * len(data) / len(listdir(train_path)), loss.item()))

        batch_id = batch_id + 1

def test():
    model.eval()
    files = listdir("data/test/")
    f = random.choice(files)
    img = Image.open("data/test/" + f)
    img_eval_tensor = transform(img)
    img_eval_tensor.unsqueeze_(0)
    data = Variable(img_eval_tensor.cuda())
    out = model(data)
    #print('Decision :', out.data.max(1, keepdim=True)[1])
    print('Decision :', out)
    img.show()
    x = input('')

if not trained_model:
    for epoch in range(1,31):
        train(epoch)

    torch.save(model, 'Catsanddogs_model.pth')

else:
    test()


