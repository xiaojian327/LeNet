import torch
from torch import nn
from net import MyLeNet5
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os

#数据转化为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

#加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=False)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
#加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=False)
test_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

#如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else 'cpu'

#调用net里面定义的模型，将模型数据转到GPU
model = MyLeNet5().to(device)

#定义一个损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()

#定义一个优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

#学习率每隔10轮，变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        #前向传播
        x, y = x.to(device), y.to(device)
        output = model(x)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)

        cur_acc = torch.sum(y == pred)/output.shape[0]

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
    print("train_loss" + str(loss/n))
    print("train_acc" + str(current/n))

def val(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            # 前向传播
            x, y = x.to(device), y.to(device)
            output = model(x)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
    print("val_loss" + str(loss / n))
    print("val_acc" + str(current / n))

    return current/n

#开始训练
epoch = 200
min_acc = 0
for t in range(epoch):
    print(f'epoch{t+1}\n..............')
    train(train_dataloader, model, loss_fn, optimizer)
    a = val(test_dataloader, model, loss_fn)
    #保存最好的模型权重
    if a > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = 0
        print('save best model')
        torch.save(model.state_dict(), 'save_model/best_model.pth')
print('Done!')
