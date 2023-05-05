import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# from models.vgg16 import VGG16
from utils.dataloader import get_cifar10_dataloaders
import torchvision
from models import VGG16, NET, AlexNet, ResNet50, EfficientNet, GoogLeNet, MobileNetV2
import argparse

# ハイパーパラメータの読み込み
with open('config/hyperparameters.json') as f:
    hyperparameters = json.load(f)

# 引数の定義
parser = argparse.ArgumentParser(description='This is a program for processing data.')
parser.add_argument('-m', '--model_name', type=str, default='resnet', help='Input model name')
args = parser.parse_args()

model_dict = {
    'normal':NET(),
    'vgg16': VGG16(),
    'AlexNet': AlexNet(),
    'resnet': ResNet50(),
    'efficient':EfficientNet("b0"),
    'googlenet':GoogLeNet(),
    'mobilenet':MobileNetV2(),
    # 他のモデルもここに追加できます。例： 'resnet50': ResNet50
}

model_name = args.model_name  # ここを変更して使用するモデルを選びます
model_class = model_dict[model_name]


# モデルの読み込み
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = model_class
net.load_state_dict(torch.load(f'./weights/{model_name}/{model_name}_cifar10.pth'))
net.to(device)
net.eval()

_, testloader = get_cifar10_dataloaders()




# テストの実行
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
