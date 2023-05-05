import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from models import VGG16, NET, AlexNet, ResNet50, EfficientNet, GoogLeNet, MobileNetV2
# from utils.dataloader import get_cifar10_dataloaders

# NumPy、Matplotlib、PyTorchをインポートする
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
from utils.dataloader import get_cifar10_dataloaders, get_mnist_dataloaders
import os
import datetime
import argparse

# 引数の定義
parser = argparse.ArgumentParser(description='This is a program for processing data.')
parser.add_argument('-m', '--model_name', type=str, default='resnet', help='Input model name')
args = parser.parse_args()

# ハイパーパラメータの読み込み
with open('config/hyperparameters.json') as f:
    hyperparameters = json.load(f)

model_dict = {
    'normal':NET(),
    'vgg16': VGG16(),
    'AlexNet': AlexNet(),
    'resnet': ResNet50(),
    'efficient':EfficientNet("b0"),
    'googlenet':GoogLeNet(),
    'mobilenet':MobileNetV2()
    # 他のモデルもここに追加できます。例： 'resnet50': ResNet50
}

model_name = args.model_name  # ここを変更して使用するモデルを選びます
model_class = model_dict[model_name]


# モデルの作成
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
net = model_class.to(device)

# 損失関数と最適化アルゴリズムの定義
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=hyperparameters['lr'])
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

trainloader, _ = get_cifar10_dataloaders()


# トレーニングの実行
for epoch in range(hyperparameters['num_epochs']):
    running_loss = 0.0
    print("start")
    print("--------")
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('Finished Training')

current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = f'./weights/{model_name}'
os.makedirs(save_dir, exist_ok=True)
# Now save your weights
# torch.save(net.state_dict(), f'{save_dir}/{model_name}_cifar10_{current_time}.pth')
torch.save(net.state_dict(), f'{save_dir}/{model_name}_cifar10.pth')
# epoch数とか入れていいはず