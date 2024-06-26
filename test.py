import torch
from net import MyLeNet5
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

#数据转化为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

#加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
#加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=False)

#如果有显卡，可以转到GPU
device = "cuda" if torch.cuda.is_available() else 'cpu'

#调用net里面定义的模型，将模型数据转到GPU
model = MyLeNet5().to(device)

#调用net里面定义的模型，将模型数据转到gpu
model = MyLeNet5().to(device)

model.load_state_dict(torch.load('save_model/best_model.pth'))

#获取结果
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

#把tensor转化为图片，方便可视化
show = ToPILImage()

#进入验证
for i in range(20):
    X, y = test_dataset[i][0], test_dataset[i][1]
    show(X).show()

    X = Variable(torch.unsqueeze(X, dim=0).float(), requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(X)
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'predicted:"{predicted}", actual:"{actual}"')