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
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
# from sklearn.model_selection import StratifiedGroupKFold

from utils.logger import Logger
from tqdm import tqdm
from dataloaders.pic_data import ImgDataset
from network import ViT

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
parser = argparse.ArgumentParser(description='CPR-HMIL')
parser.add_argument('--lr',type=float, default=1e-4)
parser.add_argument('--t', type=float, default=0.5)
parser.add_argument('--save_interval', type=int, default=5)
parser.add_argument('--test_fold','-f',type=int)
parser.add_argument('--batchsize',type=int,default=32)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--epoch', type=int, default=50)
args = parser.parse_args()

### write model configs here
root =  '/remote-home/duminjun/CEUS_proj'
save_root = './run'
pth_location = './Models/model_new.pth'
logger = Logger(save_root)
logger.global_step = 0
n_splits = 5
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# skf = StratifiedGroupKFold(n_splits=n_splits)

trainset = ImgDataset(split='train',input_size=224)
testset = ImgDataset(split='test',input_size=224)


trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batchsize,shuffle = True)
testloader = torch.utils.data.DataLoader(testset,batch_size=args.batchsize,shuffle = True)

# trainval_loader = {'train' : trainloader, 'valid' : validloader}

# model = ViT('B_16_imagenet1k', pretrained=True,image_size=224)
# model.fc = nn.Linear(model.fc.in_features,num_classes)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features,num_classes)
# model = nn.DataParallel(model) # 默认使用所有的device_ids，已经在生成模型时调用
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.1)

lrsch = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[7,20],gamma=0.3)
logger.auto_backup('./')

def train(trainloader, validloader, model, criterion, optimizer, lrsch, logger, args, epoch):
    model.train()
    losses = 0.
    loss_logger = 0.
    label_list =[]
    prob_list = []
    pred_list = []
    logger.update_step()
    for img, label in tqdm(trainloader,ascii=True,ncols=60):
        optimizer.zero_grad()

        outs = model(img.cuda())   # 列表参数是否要cuda？
        # print("opt tensor:",out)
        label = label.cuda()

        # if epoch>30:
        #     # 冻结部分层
        #     for name, param in model.named_parameters():
        #         if ("transformer" in name):
        #             param.requires_grad = False
        loss_batch = criterion(outs,label)
        norm_prob = torch.softmax(outs,dim=1)
        prob,pred = norm_prob.max(dim=1)
        loss_batch.backward()
        loss_logger += loss_batch.item()    # 显示全部loss
        optimizer.step()
        lrsch.step()
        label_list.extend(label.cpu().detach().tolist())
        prob_list.extend(prob.cpu().detach().tolist())
        pred_list.extend(pred.cpu().detach().tolist())
        
    loss_logger /= len(trainloader)
    print("Train loss:",loss_logger)
    log_metric('Train', label_list, prob_list, pred_list, logger,loss_logger)
    if not (logger.global_step % args.save_interval):
        logger.save(model,optimizer, lrsch, criterion)
        
def test(testloader, model, criterion, optimizer, lrsch, logger, args):
    model.eval()
    losses = 0.
    loss_logger = 0.
    label_list =[]
    prob_list = []
    pred_list = []

    for img, label in tqdm(testloader,ascii=True,ncols=60):
        with torch.no_grad():
            # print("img sum:",img.sum())
            outs = model(img.cuda())
            # print("opt tensor:",out)
        label = label.cuda()
        # print("label:",label)
        
        loss_batch = criterion(outs,label)
        loss_logger += loss_batch.item()    # 显示全部loss

        norm_prob = torch.softmax(outs,dim=1)
        prob,pred = norm_prob.max(dim=1)

        label_list.extend(label.cpu().detach().tolist())
        prob_list.extend(prob.cpu().detach().tolist())
        pred_list.extend(pred.cpu().detach().tolist())
        
    loss_logger /= len(testloader)
    print("Val loss:",loss_logger)

    acc = log_metric('Test', label_list, prob_list, pred_list, logger,loss_logger)

    return acc, model.state_dict()
        
def log_metric(prefix, target, prob, pred,logger,loss):
    cls_report = classification_report(target, pred, output_dict=True, zero_division=0)
    acc = accuracy_score(target, pred)
    # auc = roc_auc_score(target, prob)
    logger.log_scalar(prefix+'/loss',loss,print=False)
    # logger.log_scalar(prefix+'/AUC',auc,print=True)
    logger.log_scalar(prefix+'/'+'Acc', acc, print= True)
    logger.log_scalar(prefix+'/'+'Pos_precision', cls_report['weighted avg']['precision'], print= True)
    # logger.log_scalar(prefix+'/'+'Neg_precision', cls_report['0']['precision'], print= True)
    logger.log_scalar(prefix+'/'+'Pos_recall', cls_report['weighted avg']['recall'], print= True)
    # logger.log_scalar(prefix+'/'+'Neg_recall', cls_report['0']['recall'], print= True)
    logger.log_scalar(prefix+'/'+'Pos_F1', cls_report['weighted avg']['f1-score'], print= True)

    return acc

auc = 0
if args.test == True:
    finaltestset = ImgDataset(split='test')
    finaltestloader = torch.utils.data.DataLoader(finaltestset,batch_size=args.batchsize,shuffle = False,num_workers=8)
    model.load_state_dict(torch.load(pth_location, map_location='cpu'))
    test(finaltestloader,model,criterion,optimizer,lrsch,logger,args)
else:
    for i in range(args.epoch):
        print("===========Epoch:{}==============".format(i))
        train(trainloader,testloader, model,criterion,optimizer,lrsch,logger,args,i)
        score, model_save = test(testloader,model,criterion,optimizer,lrsch,logger,args)
        if score > auc:
            auc = score
            torch.save(model_save, pth_location)
