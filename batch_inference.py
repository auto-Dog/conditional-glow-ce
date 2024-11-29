import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import CIFAR10
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from PIL import Image
# from sklearn.model_selection import StratifiedGroupKFold

from utils.logger import Logger
from utils.ctScale import apply_color_transfer
from tqdm import tqdm
from dataloaders.pic_data import ImgDataset
from dataloaders.CVDcifar import CVDcifar,CVDImageNet,CVDPlace
from network import CondGlowModel
from utils.cvdObserver import cvdSimulateNet
from utils.utility import patch_split,patch_compose

dataset = 'local'
num_classes = 6

# argparse here
parser = argparse.ArgumentParser(description='COLOR-ENHANCEMENT')
parser.add_argument('--lr',type=float, default=1e-4)
parser.add_argument('--patch',type=int, default=4)
parser.add_argument('--size',type=int, default=32)
parser.add_argument('--t', type=float, default=0.5)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--test_fold','-f',type=int)
parser.add_argument('--batchsize',type=int,default=8)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--dataset', type=str, default='/work/mingjundu/imagenet100k/')
parser.add_argument("--cvd", type=str, default='protan')
# C-Glow parameters
parser.add_argument("--x_size", type=str, default="(3,32,32)")
parser.add_argument("--y_size", type=str, default="(3,32,32)")
parser.add_argument("--x_hidden_channels", type=int, default=128)
parser.add_argument("--x_hidden_size", type=int, default=32)
parser.add_argument("--y_hidden_channels", type=int, default=256)
parser.add_argument("-K", "--flow_depth", type=int, default=8)
parser.add_argument("-L", "--num_levels", type=int, default=3)
parser.add_argument("--learn_top", type=bool, default=False)
parser.add_argument("--x_bins", type=float, default=256.0)  # noise setting, to make input continues-like
parser.add_argument("--y_bins", type=float, default=256.0)
parser.add_argument("--prefix", type=str, default='K3_lownoise')
args = parser.parse_args()

args.x_size = eval(args.x_size)
args.y_size = eval(args.y_size)
print(args) # show all parameters
### write model configs here
save_root = './run'
pth_location = './Models/model_'+args.prefix+'.pth'
logger = Logger(save_root)
logger.global_step = 0
n_splits = 5
train_val_percent = 0.8

trainset = CVDImageNet(args.dataset,split='imagenet_subtrain',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
valset = CVDImageNet(args.dataset,split='imagenet_subval',patch_size=args.patch,img_size=args.size,cvd=args.cvd)
print(f'Dataset Information: Training Samples:{len(trainset)}, Validating Samples:{len(valset)}')

trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batchsize,shuffle = True)
valloader = torch.utils.data.DataLoader(valset,batch_size=args.batchsize*4,shuffle = True)

model = CondGlowModel(args)
model = model.cuda()
model.load_state_dict(torch.load(pth_location))

# lrsch = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20],gamma=0.3)
logger.auto_backup('./')

def sample_enhancement(image,model,args):
    ''' 根据给定的图片，进行颜色优化

    目标： $argmax_{c_i} p(\hat{c}|I^{cvd}c_i^{cvd})$ 

    '''
    model.eval()
    cvd_process = cvdSimulateNet(cvd_type=args.cvd,cuda=True,batched_input=True) # 保证在同一个设备上进行全部运算
    image_sample = Image.fromarray(image,mode='RGB')
    # image_sample_big = np.array(image_sample)/255.   # 缓存大图
    image_sample = image_sample.resize((args.size,args.size))
    image_sample = torch.tensor(np.array(image_sample)).permute(2,0,1).unsqueeze(0)/255.
    image_sample = image_sample.cuda()
    img_t:torch.Tensor = image_sample[0,...].unsqueeze(0)        # shape C,H,W

    img_out = img_t.clone()
    img_cvd_batch = cvd_process(img_t)
    out,nll = model(img_cvd_batch,reverse=True) # 上色
    # print(f'Mean Absolute grad: {torch.mean(torch.abs(img_t.grad))}, nll:{nll.item()}')

    ori_out_array = img_out.squeeze(0).permute(1,2,0).cpu().detach().numpy()

    recolor_out_array = out.clone()
    recolor_out_array = recolor_out_array.squeeze(0).permute(1,2,0).cpu().detach().numpy()
    return np.clip(recolor_out_array,0.0,1.0)*255
    # recolor_out_array_big = apply_color_transfer(ori_out_array,recolor_out_array,image_sample_big)  # 将小图变换应用到大图

    # img_out_array = img_t.clone()
    # img_out_array = img_out_array.squeeze(0).permute(1,2,0).cpu().detach().numpy()
    # img_out_array_big = apply_color_transfer(ori_out_array,img_out_array,image_sample_big)

    # img_diff = (img_out_array - ori_out_array)*10.0
    # img_diff_big = (img_out_array_big - image_sample_big)*10.0
    # img_all_array = np.clip(np.hstack([ori_out_array,recolor_out_array,img_out_array,img_diff]),0.0,1.0)
    # img_all_array_big = np.clip(np.hstack([image_sample_big,img_out_array_big,img_diff_big]),0.0,1.0)
    # plt.imshow(img_all_array)
    # plt.savefig('./run/'+f'sample_{args.prefix}_e{epoch}.png')
    # plt.cla()
    # plt.imshow(img_all_array_big)
    # plt.savefig('./run/'+f'highres_sample_{args.prefix}_e{epoch}.png')

def render_ball(color=(0,0,0),):
    # 创建一个灰色背景的角落
    normalize_color = (color[0]/255., color[1]/255., color[2]/255.)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor((60/255, 60/255, 60/255))
    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    # # 创建三个灰色平面搭建角落
    # 二元函数定义域平面
    x = np.linspace(0, 12, 12)
    y = np.linspace(0, 12, 12)
    X, Y = np.meshgrid(x, y)
    # -------------------------------- 绘制 3D 图形 --------------------------------
    # 设置X、Y、Z面的背景是白色
    ax.w_xaxis.set_pane_color((0.4,0.4,0.4, 1.0))
    ax.w_yaxis.set_pane_color((0.8,0.8,0.8, 1.0))
    ax.w_zaxis.set_pane_color((0.3,0.3,0.3, 1.0))

    # 创建一个指定颜色的球体
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 5 * np.outer(np.cos(u), np.sin(v)) + 5  # 球体中心在(5, 5, 5)
    y = 5 * np.outer(np.sin(u), np.sin(v)) + 5
    z = 5 * np.outer(np.ones(np.size(u)), np.cos(v)) + 5
    ax.plot_surface(x, y, z, color=normalize_color, linewidth=0)  # 红色球体

    # 设置坐标轴标签
    ax.set_xlim([0,11])
    ax.set_ylim([0,11])
    ax.set_zlim([0,11])
    # ax.set_aspect(1)

    # 显示图形
    # plt.show()
    plt.savefig('tmp.png')
    image_back = Image.open('tmp.png').convert('RGB')
    return np.array(image_back)

def render_patch(color=(0,0,0)):
    ''' 生成一个400x600的图像，中心色块为指定颜色。背景颜色为128,128,128'''
    # 创建一个400x600的图像，背景颜色为128,128,128
    image = np.full((400, 600, 3), 128, dtype=np.uint8)

    # 获取色块的位置
    start_x = 250  # 中心位置向左偏移
    start_y = 150  # 中心位置向上偏移

    # 用户指定的色块RGB值
    color_rgb = color  # 示例红色

    # 在图像上绘制色块
    image[start_y:start_y+100, start_x:start_x+100] = color_rgb

    # 为色块添加2像素黑色边缘
    image[start_y:start_y+2, start_x:start_x+100] = (0, 0, 0)
    image[start_y+98:start_y+100, start_x:start_x+100] = (0, 0, 0)
    image[start_y:start_y+100, start_x:start_x+2] = (0, 0, 0)
    image[start_y:start_y+100, start_x+98:start_x+100] = (0, 0, 0)

    # # 显示图像
    # plt.imshow(image)
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()
    # pass
    return image

# 产生指定颜色值
colors = []
for r in range(0, 256, 10):
    for g in range(0, 256, 10):
        for b in range(0, 256, 10):
            colors.append((r, g, b))
colors_out = []
for rgb_value_ori in tqdm(colors):
    # 产生指定颜色值色块
    img_patch = render_patch(rgb_value_ori)
    img_patch = render_ball(rgb_value_ori)
    # 进行色盲模拟和重新上色
    img_recolor = sample_enhancement(img_patch,model,args)
    # 取img_recolor[15:17,15:17]并求平均
    rgb_recolor_value = np.mean(img_recolor[15:17,15:17,:],axis=(0, 1))
    colors_out.append(rgb_recolor_value)
colors = np.array(colors)
colors_out = np.array(colors_out)
colors_out = np.hstack([colors,colors_out])
np.savetxt( "colormap_sphere.csv", colors_out, delimiter=",") # 保存结果


