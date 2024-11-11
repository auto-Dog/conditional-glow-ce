import os
import random
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
# from model import Classifier
import torchvision.models as models
import matplotlib.pyplot as plt

#------超参数和其他常用变量控制台 AlexNet------#
proj_name = 'vggmininodule'
batch_size = 32
num_epoch = 40
learning_rate = 0.0001
num_of_class = 6
input_size0 = 224
input_size1 = 224
myrootdir_pics = './dataset'    #测试集和训练集所在地址
pth_tar_location = './model_minivgg_'+proj_name+'.pth.tar'   #保存训练结果文件
#---------------------------------------------#
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 读取数据集文件(黑白);要使用彩色请看注释
# 文件名格式为id，要放在对应类别的文件夹下(0,1,2...)
def readfile(path):
    j=0
    file_count=0
    for dirpath, dirnames, filenames in os.walk(path):
        for file in filenames:
            file_count=file_count+1
    class_folders = sorted(os.listdir(path))
    print("\n Total %d classes and %d images "%(len(class_folders),file_count))
    # x = np.zeros((file_count, input_size0, input_size1), dtype=np.uint8)    # 黑白图片
    x = np.zeros((file_count, input_size0, input_size1,3), dtype=np.uint8)    # 彩色图片
    y = np.zeros((file_count), dtype=np.uint8)
    for sub_folders in class_folders:
        image_dir=os.listdir(os.path.join(path,sub_folders))
        # image_dir.sort(key= lambda x:int(x[:-4]))
        for i, file_name in enumerate(image_dir):
            # img = Image.open(os.path.join(path,sub_folders,file_name)).convert('L')   # 或使用opencv                
            # x[j+i, :] = np.array(img.resize((input_size0, input_size1)),dtype=np.uint8) # 或使用opencv

            # img = cv2.imread(os.path.join(path,sub_folders,file_name),0)    # 黑白图片
            # x[j+i, :] = cv2.resize(img, (input_size0, input_size1))    # 黑白图片

            img = cv2.imread(os.path.join(path,sub_folders,file_name))    # 彩色图片
            x[j+i, :, :] = cv2.resize(img, (input_size0, input_size1))   # 彩色图片
            y[j+i] = int(eval(sub_folders))
        j+=(i+1)
    print(y.shape)
    return x, y


# 设置随机数种子
setup_seed(1896)

# 分別将 training set、validation set 用 readfile 函数读进来;
workspace_dir = myrootdir_pics
print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"))
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "testing"))
print("Size of validation data = {}".format(len(val_x)))

# 训练集图片预处理
# ToTensor: Convert a PIL image(H, W, C) in range [0, 255] to
#           a torch.Tensor(C, H, W) in the range [0.0, 1.0]
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# 验证集图片预处理
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


# 构建训练集、验证集;
class ImgDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        # label is required to be a LongTensor
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        X = self.transform(X)
        Y = self.y[index]
        return X, Y


train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, val_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# 检测是否能够使用GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 搭建模型 model;
# 构造损失函数 loss;
# 构造优化器 optimizer;
# 设定训练次数 num_epoch;

# (1) 使用自定义模型，在model.py中修改
# model = Classifier().to(device)
#当要指定分类数目，采用这一句代替上面一行语句
# model = Classifier(num_of_class).to(device)
# model = models.densenet169()
# # 修改最后一层全连接层输出的种类数
# model.classifier = torch.nn.Linear(model.classifier.in_features, num_of_class)
# model = model.to(device)

#（2）使用pytorch官方模型:
model = models.resnet18(pretrained = True)
model.fc = nn.Linear(model.fc.in_features,num_of_class)
model = model.to(device)
if torch.cuda.device_count() > 1:   # 多gpu训练
    print('Multi-GPU detected')
    model = nn.DataParallel(model)
loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate)  # optimizer 使用 Adam
val_acc_best = 0.0
result_plots = np.zeros((4,num_epoch))
x_plots = range(1,num_epoch+1)
# 训练 并print每个epoch的结果;
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()  # 确保 model 是在 train model (开启 Dropout 等...)
    for i, data in enumerate(train_loader): # data[0]: 输入图片，np数组, data[1]: label，0，1，2...数字
        optimizer.zero_grad()  # 用 optimizer 将 model 参数的 gradient 归零
        train_pred = model(data[0].to(device))  # 调用 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].to(device))  # 计算 loss
        batch_loss.backward()  # 利用 back propagation 算出每个参数的 gradient
        optimizer.step()  # 以 optimizer 用 gradient 更新参数值

        train_acc += np.sum(
            np.argmax(train_pred.cpu().data.numpy(), axis=1) ==
            data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].to(device))
            batch_loss = loss(val_pred, data[1].to(device))

            val_acc += np.sum(
                np.argmax(val_pred.cpu().data.numpy(), axis=1) ==
                data[1].numpy())
            val_loss += batch_loss.item()

    train_acc /= train_set.__len__()
    train_loss /= train_set.__len__()
    val_acc /= val_set.__len__()
    val_loss /= val_set.__len__()

    # 将结果 print 出来
    print(
        '[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f'
        % (epoch + 1, num_epoch, time.time() - epoch_start_time, train_acc,
           train_loss, val_acc, val_loss))
    # 记录中间结果
    result_plots[0,epoch] =  train_acc
    result_plots[1,epoch] =  train_loss
    result_plots[2,epoch] =  val_acc
    result_plots[3,epoch] =  val_loss


    # 记录最好的结果 并保存模型
    if val_acc > val_acc_best:
        val_acc_best = val_acc
        torch.save(model.state_dict(), pth_tar_location)
        print('Save model')

#绘制结果
print('Best accuracy on validation set: %3.6f' % val_acc_best)
plt.title('Acc and loss')
plt.subplot(1,2,1)
plt.plot(x_plots,result_plots[0,:],"-r",label='train_acc')
plt.plot(x_plots,result_plots[2,:],"-b",label='val_acc')
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_plots,result_plots[1,:],"-m",label='train_loss')
plt.plot(x_plots,result_plots[3,:],"-c",label='val_loss')
plt.legend()
plt.savefig('./res_'+proj_name+'.png')
plt.show()
