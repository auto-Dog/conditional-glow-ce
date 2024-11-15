import numpy as np
import sys
import torch
import torch.nn as nn

class colorConverter(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ct_conv_1 = nn.Conv2d(3,32,(1,1))
        self.ct_conv_2 = nn.Conv2d(32,32,(1,1))
        self.ct_conv_3 = nn.Conv2d(32,3,(1,1))
        self.relu = nn.ReLU()
    
    def forward(self,x):
        out = self.ct_conv_1(x)
        out = self.relu(out)
        out = self.ct_conv_2(out)
        out = self.relu(out)
        out = self.ct_conv_3(out)
        return out

if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    cc_model = colorConverter()
    image_sample = Image.open('../apple.png').convert('RGB').resize((32,32))
    image_sample = torch.tensor(np.array(image_sample)).permute(2,0,1).unsqueeze(0)/255.
    # image_sample = image_sample.cuda()
    img_t:torch.Tensor = image_sample[0,...].unsqueeze(0)        # shape C,H,W
    img_out = img_t.clone()
    inference_optimizer = torch.optim.SGD(params=list(cc_model.parameters()),lr=1)   # 对输入图像进行梯度下降
    cc_model.train()
    criteria = nn.MSELoss()
    for iter in range(1000):
        inference_optimizer.zero_grad()
        img_t_new = cc_model(img_t) # 调色
        loss = criteria(img_t_new,img_t)
        loss.backward()
        inference_optimizer.step()

    ori_out_array = img_t.squeeze(0).permute(1,2,0).detach().numpy()
    img_out_array = img_t_new.clone()
    img_out_array = img_out_array.squeeze(0).permute(1,2,0).detach().numpy()
    img_diff = (img_out_array != ori_out_array)*1.0
    img_out_array = np.clip(np.hstack([ori_out_array,img_out_array]),0.0,1.0)
    plt.imshow(img_out_array)
    plt.show()
    torch.save(cc_model.state_dict(),'equ_transform.pth')
    # plt.savefig('./run/'+f'sample_cc_e{epoch}.png')