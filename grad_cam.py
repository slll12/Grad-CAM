import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch.nn.functional as F
from model import Model
from torchvision import transforms as T

### 流程
# 入口是main函数

# 1. 使用pytorch的hook机制获得梯度图和特征图
# 2. 梯度图全局平均池化获得加权权重，按通道维度将得到权重对特征图进行加权求和

def get_model(check_point):

    ## 实例化模型
    mymodel = Model()

    ## 载入模型权重
    # mymodel.load_state_dict(torch.load(check_point, map_location="cpu"))
    mymodel=torch.load(check_point, map_location="cpu")
    print("ALL KEys")
    ##固定dropout和normalize
    mymodel.eval()
    return mymodel


def get_transform():
    class ThresholdTransform(object):
        def __init__(self, thr_255):
            self.thr = thr_255 / 255.

        def __call__(self, x):
            return (x > self.thr).to(x.dtype)
    transform = T.Compose([
        T.Resize((28, 28)),
        T.ToTensor(),
        ThresholdTransform(100.),
    ])
    return transform


## grad_out->[1,C,H,W]
def backward_hook(module, grad_in, grad_out):
    ## module :显示cam的模块
    ## grad_in : 模块的输入
    ## grad_out : 模块的梯度

    ## 梯度图全局平均获得加权权重
    ### grad_out-＞[1,C,H,W]
    ### weight->[1,C,H,W]
    weight = F.adaptive_max_pool2d(grad_out[0], 1)
    grad_block.append(weight[0].detach())


def farward_hook(module, input, output):
    feature_block.append(output[0])


def get_cam(imgpath, mymodel, transform,mode='RGB'):
    if mode=='RGB':
        a = Image.open(imgpath).convert(mode)
    else:
        ## 样例是单通道，根据自己数据确定是'L'还是'RGB'
        a=Image.open(imgpath).convert('L')

    ## 原图片用于展示
    raw_img=np.array(a.convert('RGB'))

    # a->[1,3,H,W]
    a = torch.unsqueeze(transform(a), 0)

    output = mymodel(a)
    print(output.shape)

    mymodel.zero_grad()

    max_idx = np.argmax(output.cpu().data.numpy())
    class_loss = output[0,max_idx]
    class_loss.backward()

    # grads_map->[C,1,1]    fmap->[C,H,W]
    grads_map = grad_block[0]
    fmap = feature_block[0]

    ## 加权求和，relu抑制负值
    cam = (grads_map * fmap).sum(0)
    cam = torch.relu(cam)

    ## 归一化，转rgb,便于显示
    cam = cam.cpu().data.numpy().squeeze()
    cam = cam / cam.max()
    cam = cv2.resize(cam, (raw_img.shape[1],raw_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap=cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)

    ## 原图和cam图叠加显示，比例可以自己调
    cam_img = 0.5* heatmap + 0.5 * raw_img
    return cam_img / 255


if __name__ == '__main__':
    ## 修改成自己的模型和权重*
    mymodel = get_model(check_point='Lenet.pth')

    ## 修改成test的transform*
    transform = get_transform()

    ## 存梯度图，存特征图
    grad_block = []
    feature_block = []

    # hook函数，勾选要显示的模块(这里勾选的是LeNet的self.p1),一般是最后一个卷积块的pooling层(卷积作为自动特征提取器获得优良特征供mlp进行分类)
    # 设置正向传播自动调用farward_hook,反向自动调用backward_hook *
    mymodel.p1.register_forward_hook(farward_hook)
    mymodel.p1.register_backward_hook(backward_hook)

    cam=get_cam(imgpath='test.png', mode='L',transform=transform,mymodel=mymodel)
    plt.imshow(cam)
    plt.show()
