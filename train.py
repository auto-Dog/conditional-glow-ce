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
from PIL import Image
# from sklearn.model_selection import StratifiedGroupKFold

from utils.logger import Logger
from tqdm import tqdm
from dataloaders.pic_data import ImgDataset
from dataloaders.CVDcifar import CVDcifar,CVDImageNet,CVDPlace
from network import CondGlowModel
from utils.cvdObserver import cvdSimulateNet
from utils.conditionP import conditionP
from utils.utility import patch_split,patch_compose

# hugface官方实现
# from transformers import ViTImageProcessor, ViTForImageClassification
# processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits

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
parser.add_argument('--batchsize',type=int,default=256)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--epoch', type=int, default=50)
# C-Glow parameters
parser.add_argument("--x_size", type=tuple, default=(3,32,32))
parser.add_argument("--y_size", type=tuple, default=(3,32,32))
parser.add_argument("--x_hidden_channels", type=int, default=128)
parser.add_argument("--x_hidden_size", type=int, default=32)
parser.add_argument("--y_hidden_channels", type=int, default=256)
parser.add_argument("-K", "--flow_depth", type=int, default=8)
parser.add_argument("-L", "--num_levels", type=int, default=3)
parser.add_argument("--learn_top", type=bool, default=False)
parser.add_argument("--x_bins", type=float, default=256.0)  # noise setting, to make input continues-like
parser.add_argument("--y_bins", type=float, default=256.0)
args = parser.parse_args()

### write model configs here
save_root = './run'
pth_location = './Models/model_new.pth'
logger = Logger(save_root)
logger.global_step = 0
n_splits = 5
train_val_percent = 0.8
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# skf = StratifiedGroupKFold(n_splits=n_splits)

# trainset = CVDcifar('./',train=True,download=True,patch_size=args.patch)
# testset = CVDcifar('./',train=False,download=True,patch_size=args.patch)
# trainset = CVDImageNet('/kaggle/input/imagenet1k-subset-100k-train-and-10k-val',split='imagenet_subtrain',patch_size=args.patch,img_size=args.size)
# valset = CVDImageNet('/kaggle/input/imagenet1k-subset-100k-train-and-10k-val',split='imagenet_subval',patch_size=args.patch,img_size=args.size)
trainset = CVDPlace('/work/mingjundu/place_dataset/places365_standard/',split='train',patch_size=args.patch,img_size=args.size)
valset = CVDPlace('/work/mingjundu/place_dataset/places365_standard/',split='val',patch_size=args.patch,img_size=args.size)
# inferenceset = CIFAR10('./',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),]))

# train_size = int(len(trainset) * train_val_percent)   # not suitable for ImageNet subset
# val_size = len(trainset) - train_size
# trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
print(f'Dataset Information: Training Samples:{len(trainset)}, Validating Samples:{len(valset)}')

trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batchsize,shuffle = True)
valloader = torch.utils.data.DataLoader(valset,batch_size=args.batchsize,shuffle = True)
# testloader = torch.utils.data.DataLoader(testset,batch_size=args.batchsize,shuffle = False)
# inferenceloader = torch.utils.data.DataLoader(inferenceset,batch_size=args.batchsize,shuffle = False,)
# trainval_loader = {'train' : trainloader, 'valid' : validloader}

# model = ViT('ColorViT', pretrained=False,image_size=32,patches=4,num_layers=6,num_heads=6,num_classes=4*4*3)
model = CondGlowModel(args)
# model = nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count())))
model = model.cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.1)

lrsch = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20],gamma=0.3)
logger.auto_backup('./')

def train(trainloader, model, criterion, optimizer, lrsch, logger, args, epoch):
    model.train()
    loss_logger = 0.
    logger.update_step()
    for img, ci_patch, img_target, ci_rgb in tqdm(trainloader,ascii=True,ncols=60):
        optimizer.zero_grad()
        img = img.cuda()
        img_target = img_target.cuda()
        outs,nll = model(img+torch.rand_like(img) / args.x_bins,
                         img_target+torch.rand_like(img_target) / args.y_bins) 
        # img_target = img_target.cuda()
        # print("opt tensor:",out)
        # ci_rgb = ci_rgb.cuda()

        # if epoch>30:
        #     # 冻结部分层
        #     for name, param in model.named_parameters():
        #         if ("transformer" in name):
        #             param.requires_grad = False
        # loss_batch = criterion(outs,img_target)
        loss_batch = torch.mean(nll)
        loss_batch.backward()
        loss_logger += loss_batch.item()    # 显示全部loss
        optimizer.step()
        lrsch.step()

    loss_logger /= len(trainloader)
    print("Train loss:",loss_logger)
    log_metric('Train', logger,loss_logger)
    if not (logger.global_step % args.save_interval):
        logger.save(model,optimizer, lrsch, criterion)
        
def validate(testloader, model, criterion, optimizer, lrsch, logger, args):
    model.eval()
    loss_logger = 0.

    for img, ci_patch, img_target, ci_rgb in tqdm(testloader,ascii=True,ncols=60):
        with torch.no_grad():
            outs,nll = model(img.cuda(),img_target.cuda()) 
        # ci_rgb = ci_rgb.cuda()
        # img_target = img_target.cuda()
        # print("label:",label)
        
        # loss_batch = criterion(outs,img_target)
        loss_batch = torch.mean(nll)
        loss_logger += loss_batch.item()    # 显示全部loss

    loss_logger /= len(testloader)
    print("Val loss:",loss_logger)

    acc = log_metric('Val', logger,loss_logger)

    return acc, model.state_dict()

def sample_enhancement(model,inferenceloader,epoch):
    ''' 根据给定的图片，进行颜色优化

    目标： $argmax_{c_i} p(\hat{c}|I^{cvd}c_i^{cvd})$ 

    '''
    model.eval()
    cvd_process = cvdSimulateNet(cuda=True,batched_input=True) # 保证在同一个设备上进行全部运算
    # for img,_ in inferenceloader:
    #     img = img.cuda()
    #     img_cvd = cvd_process(img)
    #     img_cvd:torch.Tensor = img_cvd[0,...].unsqueeze(0)  # shape C,H,W
    #     img_t:torch.Tensor = img[0,...].unsqueeze(0)        # shape C,H,W
    #     break   # 只要第一张
    image_sample = Image.open('apple.png').convert('RGB').resize((64,64))
    image_sample = torch.tensor(np.array(image_sample)).permute(2,0,1).unsqueeze(0)/255.
    image_sample = image_sample.cuda()
    img_cvd = cvd_process(image_sample)
    img_cvd:torch.Tensor = img_cvd[0,...].unsqueeze(0)  # shape C,H,W
    img_t:torch.Tensor = image_sample[0,...].unsqueeze(0)        # shape C,H,W

    img_out = img_t.clone()
    # inference_criterion = nn.MSELoss()
    img_t.requires_grad = True
    inference_optimizer = torch.optim.SGD(params=[img_t],lr=args.lr*1000,momentum=0.3)   # 对输入图像进行梯度下降
    for iter in range(100):
        inference_optimizer.zero_grad()
        img_cvd_batch = cvd_process(img_t)
        out,loss = model(img_cvd_batch,img_out)  # 相当于-log p(img_ori|img_cvd(t))
        # loss = inference_criterion(out,img_out)   
        loss.backward()
        inference_optimizer.step()
        if iter%10 == 0:
            print(f'Mean Absolute grad: {torch.mean(torch.abs(img_t.grad))}')

    # img_out = img_t.clone()
    # inference_criterion = conditionP()
    # img_cvd_batch = img_cvd.repeat(64,1,1,1)
    # img_t_patches = patch_split(img_t.clone())
    # img_t_patches.requires_grad = True
    # img_ori_patches = patch_split(img_t)
    # inference_optimizer = torch.optim.SGD(params=[img_t_patches],lr=args.lr,momentum=0.3)   # 对输入图像进行梯度下降
    # for iter in range(30):
    #     inference_optimizer.zero_grad()
    #     img_cvd_patches = cvd_process(img_t_patches)
    #     out = model(img_cvd_batch,img_cvd_patches)
    #     loss = inference_criterion(out,img_ori_patches)    # 相当于-log p(img_ori_patch|img_cvd,img_t_patch)
    #     loss.backward()
    #     inference_optimizer.step()

    ori_out_array = img_out.squeeze(0).permute(1,2,0).cpu().detach().numpy()

    recolor_out_array = out.clone()
    recolor_out_array = recolor_out_array.squeeze(0).permute(1,2,0).cpu().detach().numpy()

    img_out_array = img_t.clone()
    img_out_array = img_out_array.squeeze(0).permute(1,2,0).cpu().detach().numpy()
    img_diff = (img_out_array != ori_out_array)*1.0
    img_out_array = np.clip(np.hstack([ori_out_array,recolor_out_array,img_out_array,img_diff]),0.0,1.0)
    plt.imshow(img_out_array)
    plt.savefig('./run/'+f'sample_e{epoch}.png')



def log_metric(prefix, logger, loss):
    logger.log_scalar(prefix+'/loss',loss,print=False)
    return 1/loss   # 越大越好

testing = validate
auc = 0

if args.test == True:
    finaltestset = CVDcifar('./',train=False,download=True)
    finaltestloader = torch.utils.data.DataLoader(finaltestset,batch_size=args.batchsize,shuffle = False,num_workers=8)
    model.load_state_dict(torch.load(pth_location, map_location='cpu'))
    sample_enhancement(model,None,-1)
    # testing(finaltestloader,model,criterion,optimizer,lrsch,logger,args)
else:
    for i in range(args.epoch):
        print("===========Epoch:{}==============".format(i))
        # sample_enhancement(model,inferenceloader,i) # debug
        train(trainloader, model,criterion,optimizer,lrsch,logger,args,i)
        score, model_save = validate(valloader,model,criterion,optimizer,lrsch,logger,args)
        sample_enhancement(model,None,i)
        if score > auc:
            auc = score
            torch.save(model_save, pth_location)
