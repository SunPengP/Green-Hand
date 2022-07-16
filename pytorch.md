```python
torch.nn.Parameter是继承自torch.Tensor的子类，其主要作用是作为nn.Module中的可训练参数使用,
nn.Parameter会自动被认为是module的可训练参数nn.Parameter的对象的requires_grad属性的默认值是True，即是可被训练的。

self.register_parameter('q_proj_weight', None) # 向我们建立的网络module添加 parameter


torch.nn.init.xavier_uniform_(tensor, gain=1)
xavier初始化方法中服从均匀分布U(−a,a) ，分布的参数a = gain * sqrt(6/fan_in+fan_out)，
这里有一个gain，增益的大小是依据激活函数类型来设定

```

![image-20220327203958083](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20220327203958083.png)

```python
args = parser.parse_args() # 解析参数
```

```python
import os.path # 主要用于获取文件的属性
```

```python
CSV是一种非常简单的数据格式
csv模块实现用于以 CSV 格式读取和写入表格数据的类。 csv模块的reader和writer对象读取和写入序列。
```



```pyrhon
pytorch 加载图像数据集需要两步，首先需要使用torchvision.dataset.ImageFolder()读取图像，
然后再使用torch.utils.data.DataLoader()加载数据集
```



# 在cmd创建虚拟环境

``` python
# 在名为42的文件夹的项目里新建了虚拟环境py29

conda create -n 虚拟环境名字 python==3.8.5
#创建完成后，如果要使用，需要将虚拟环境激活
conda activate 虚拟环境名字

# 标签文件的名称和图片的名称应该是
```



## np.array 和tensor的相互转换

```python
import torch
import numpy as np
x = np.ones(5)
x = torch.tensor(x)
```

## pytorch的dim和numpy的axis

```pyhton
对于一个张量，他的shape有几维，就对应几个轴
dim和axis的效果一样
```

![image-20220401200421270](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20220401200421270.png)

![image-20220401200438338](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20220401200438338.png)

- 

```python
torch.meshgrid(a,b) # 生成网格，用于生成坐标
```

![image-20220401202817804](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20220401202817804.png)



```python
torch.bmm(a,b) # 计算两个tensor的矩阵乘法
```

### argparse --命令行选项、参数和子命令解析器

[`argparse`](https://docs.python.org/zh-cn/3/library/argparse.html#module-argparse) 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的参数，然后argparse将弄清如何从sys.argv解析出那些参数。还会自动生成帮助和使用手册。

```python
# 获取一个整数列表并计算总和或者最大值
import argparse
# 创建一个解析器
parser = argparse.ArgumentParser(description='Process some integers')
# add_argument()方法添加参数
parser.add_argument('integers',metavar='N',type=int,nargs='+',help='an integer for the accumulator')
# 在命令行中指定了--sum参数时将是sum()函数。否则默认是max()函数
parser.add_argument('--sum',dest='accumulate',action='store_const',const=sum,default=max,
                   help='sum the integers(default:find the max)')
parser.add_argument('--foo', action='store_const', const=42)
# 对添加的参数进行存储和使用
args = parser.parse_args(['7','-1','42'])
print(args.accumulate(args.integers))
```

### F.interpolate

```python
# F.interpolate()上下采样操作
# torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
# size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]) –输出大小
# scale_factor (float or Tuple[float]) – 指定输出为输入的多少倍数
import torch
import torch.nn as nn
import torch.nn.functional as F
input = torch.arange(1,5,dtype=torch.float32).view(1,1,2,2)
x = F.interpolate(input,scale_factor=2,mode='nearest') 
print(x)
x = F.interpolate(input,scale_factor=2,mode='bilinear',align_corners=True) # 双线性插值
print(x)
```

![](C:\Users\lenovo\Desktop\noteOf_languageAnd_tools\PyTorch_learnNotebook\Python_Debug_Image\inter.png)



![](C:\Users\lenovo\Desktop\noteOf_languageAnd_tools\PyTorch_learnNotebook\Python_Debug_Image\inter2.png)

```python
import torch
a = torch.ones(2,3,6)
b = a.shape[:2] # 返回2，3
a.flatten(0,1).shape # [2*3,6]

# softmax(input,dim=None) dim常用0，1，2，-1
# dim=0  是对每一维度相同位置的数值进行softmax运算
# 当dim=1时， 是对某一维度的列进行softmax运算
# 当dim=2或-1时， 是对某一维度的行进行softmax运算
torch.cdis(a,b,p) # p可以为0，1，2 分别代表L0,L1,L2范数
```



##### 关于cumsum

![cumsum](C:\Users\lenovo\Desktop\PyTorch语法查询笔记\Python_Debug_Image\cumsum.PNG)

![image-20220408153602656](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20220408153602656.png)



##### nn.Linear()用于设置网络中的全连接层，在二维图像处理的任务中，全连接层的输入输出一般都设置为二维张量，形状通常为[batch_size,size]

```python
torch.nn.Linear(in_features,out_features,bias=True)

# 使用_get_clones()方法将结构相通的层复制
_get_clones(encoder_layer, num_layers)
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
```

![image-20220408181231989](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20220408181231989.png)



```python
# torch.full(size,fill_value)  /   torch.full_like(input,fill_value)
```

![image-20220409095753143](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20220409095753143.png)

```python
# expand函数
torch.expand(int,-1) # -1表示该位置的维度保持不变
```



##### ViT的使用

![image-20220417142329663](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20220417142329663.png)



#### 正太分布初始化

```python
import torch
torch.normal(0,0.01,size,require_grad=True)
```



```python
model.children # 提取model每一层的网络结构
```



```python
import troch
torch.linspace(start,end,step,out) #返回一个1维张量，包含在区间start和end上均匀间隔的step个点, 输出的张量的长度由steps决定

for a,b in enumerate() # a 位置上是索引，b位置上是内容
```



#### nn.ModuleList和nn.Sequential

```python
nn.ModuleList,它是一个存储不同module，并自动将每个module的parameters添加到网络之中的容器。你可以把任意nn.Module的子类（如nn.Conv2d，nn.Linear等）加到这个list里面，方法和python自带的list一样，无非是extend，append等操作，但不同于一般的list，加入到nn.ModuleList里面的module是会自动注册到整个网络上的，同时module的parameters也会自动添加到整个网络中。ModuleList中的内容是没有顺序的，可以按照列表的方式随意调用里面的一个module。

nn.Sequential已经实现了内部的forward函数，而且里面的模块必须是按照顺序进行排列的，所以我们必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。
```



```python
import logging # 使用python自带的logging模块实现日志功
```

```python
import os
os.listdir(root) # root 文件下的所有文件名
os.path.isdir(path) # 判断是否是目录
os.path.splitext(“文件路径”)   # 分离文件扩招名
```

```python
model.state_dict() # 打印model中所有参数名。
model.named_parameters() # 打印model中所有参数名及具体的值(包括通过继承得到的父类中的参数)
```

```python
# pytorch中学习率调整方法
lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
```



#### 在服务器上解压文件

```
1. 在服务器自己的文件夹下直接wget:  wget https://www.rarlab.com/rar/rarlinux-x64-611.tar.gz
解压:tar zxvf rarlinux-x64-611.tar.gz

2.下载后上传至服务器解压: tar -xvf rarlinux-x64-611.tar.gz
```



```python
import torch
from torch.nn.init import constant_
a = torch.ones(3,3)
constant_(a,0.1) # 用0.1填充a
```



```python
nn.Linear有weight和bias两个属性
可以通过nn.Linear.weight.data/.bias来访问
```



##### COCO API帮助加载，解析和可视化COCO中的注释。



-————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

## 2022年7月12日

##### torch.randint(low,high,size)

```python
import torch
a = torch.randint(0,255,(3,2,2) # 随机生成一个三通道的2*2的张量，其元素值在0~255之间
          
mask = torch.Tensor([[1,0,0],[0,1,0],[0,0,1]]).bool()#以bool形式表示mask

# 函数 masked_fill_(mask,value)：用value填充tensor中与mask中的1对应的位置
a.masked_fill_(~mask,0) # 先对mask的元素取反，则除对角线外其他位置的元素变为1，用0在a中填充与~mask为1的位置对应的位置。则最后保留了a中的所有通道对角线上的元素
              
```

##### nn.Embedding(个数，维度)

```python
import torch.nn as nn
embed = nn.Embedding(8,6) # embedding 8 个元素，每个元素embedding的维度为6，元素的取值在0~7之间
b = torch.tensor([[1,2,3],[4,5,6]]) # 对每个元素embedding
# 输出结果：tensor([[[-0.2214,  2.2463,  0.9124,  0.1492, -0.0554,  0.5254],
        # [ 0.2612,  0.0782, -0.0253, -0.8239,  0.0832, -0.1054],
        # [-0.0963, -1.4206,  0.7753, -0.9020,  0.0849,  0.1289]],

        # [[ 0.3654,  0.5165,  1.1418,  0.8297, -0.1916,  0.6715],
        # [ 0.7370, -0.2906,  1.0729,  0.0797,  0.9721,  1.0041],
        # [-1.5183, -1.1892, -0.7729, -1.1844, -0.0929, -1.1310]]], grad_fn=<EmbeddingBackward0>)
```



##### nn.Linear(in_dim,out_dim)

```python
import torch
import torch.nn as nn
a = torch.randn(2,3,3)
line = nn.Linear(9,3)
b = a.view(2,3*3) # 组后的维度要跟Linear的输入维度对应上
c = line(b)
# nn.Linear有weight和bias两个属性
# 可以通过nn.Linear.weight.data/.bias来访问
# weight的shape为(output_chinnal,input_chinnal)
# bias的shape为(output_chinnal)
```



## 7月13日

##### PyTorch中的 repeat 函数

```python
# repeat函数可以对张量进行重复扩充
#当参数只有两个时：（列的重复倍数，行的重复倍数）。1表示不重复
#当参数有三个时：（通道数的重复倍数，列的重复倍数，行的重复倍数）。
```

![](C:\Users\lenovo\Desktop\noteOf_languageAnd_tools\PyTorch_learnNotebook\Python_Debug_Image\{SSV2{NQIYWB164_J}K4ET0.png)



## 7月14日

##### PyTorch F.grid_sample

```python
# torch.nn.functional.grid_sample(input, grid, mode=‘bilinear’, padding_mode=‘zeros’, align_corners=None) # bilinear是双线性插值
# example:input_shape(N,C,H_in,W_in)
#         grid_shape(N,H_out,W_out,2)
#  then:  out_shape(N,C,H_out,W_out)
```

### nn.ModuleList()与nn.Sequential()

##### nn.ModuleList()

```python
# nn.ModuleList,它是一个存储不同module，并自动将每个module的parameters添加到网络之中的容器。
# 可以把任意nn.Module的子类（如nn.Conv2d，nn.Linear等）加到这个list里面，方法和python自带的list一样，无非是extend，append等操作，但不同于一般的list，加入到nn.ModuleList里面的module是会自动注册到整个网络上的，同时module的parameters也会自动添加到整个网络中。
# nn.ModuleList()里的顺序不会限制forward里的顺序（不影响执行顺序，但是一般设置ModuleList中的顺序和forward中保持一致，增强代码的可读性。）
# 既然ModuleList可以根据序号来调用，那么一个模型可以在forward函数中被调用多次。但需要注意的是，被调用多次的模块，是使用同一组parameters的，也就是它们是参数共享的。
```

##### nn.Sequential()

```python
# 不同于nn.ModuleList，nn.Sequential已经实现了内部的forward函数，而且里面的模块必须是按照顺序进行排列的，所以我们必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。
class net5(nn.Module):
    def __init__(self):
        super(net5, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(1,20,5),
                                    nn.ReLU(),
                                    nn.Conv2d(20,64,5),
                                    nn.ReLU())
    def forward(self, x):
        x = self.block(x)
        return x
net = net5()
print(net)
# net5(
#   (block): Sequential(
#     (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#     (1): ReLU()
#     (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#     (3): ReLU()
#   )
# )
```



## 7月15日

##### nn.Parameter()

```python
# nn.Paremeter里的参数时可以跟随网络一起训练的
```

![](C:\Users\lenovo\Desktop\noteOf_languageAnd_tools\PyTorch_learnNotebook\Python_Debug_Image\parameter.png)

```python
# named_parameters()和parameters()，前者给出网络层的名字和参数的迭代器，而后者仅仅是参数的迭代器。
```

##### 模型的所有参数都放在state_dict中

```python
net.state_dict() # 不可跟着网络一起训练的参数也存放在此处（不包括普通属性）
```

##### self.register_buffer('name',tensor) # 其中的参数不可更新

```python
# 在用self.register_buffer(‘name’, tensor) 定义模型参数时，其有两个形参需要传入。第一个是字符串，表示这组参数的名字；第二个就是tensor 形式的参数。
import torch
import torch.nn as nn 
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.register_buffer('buf1',torch.ones(2,2))
        self.register_buffer('buf2',torch.randn(2,2))
    def forward(self,x):
        ......
# 访问register_buffer参数
Model.buf1 # 用名称访问特定的register_buffer
list(Model.buffers()) # 获得所有的register_buffer
list(Model.named_buffers()) # ...,同时获取名字
```

##### nn.BatchNorm2d

```python
m = nn.BatchNorm2d(num_channel,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True) # affine设置为true表示weight和bias将被使用
# num_channel对应每个feature_map的通道数

```

##### getattr(object,属性名)

```python
class A(object):
    def set(self,a,b):
        x =a
        a =b
        b=x
        print(a,b)
    def sum(self,a,b):
        x = a+b
        print(x)
        
a = A() # 实例化一个对象
c = getattr(a,'sum') # 获取这个对象，名为‘sum’的属性
c(a=1,b=2) # c继承了该属性，等同于函数sum的操作
```

##### torchvision.models

```python
# torchvision.models里包含了一些常用的模型
import torchvision.models as tm
model = tm.resnet50(pretrained=True) # 获取预训练的resnet50,如果只需要网络结构,就设置为False
#####################################################################################

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo # model_zoo是和导入预训练模型相关的包
# __all__定义了可以从外部import 的函数名或者类名
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']
# model_urls是预训练模型的下载地址  
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
```

##### from torchvision .models._utils import IntermediateLayerGetter

```python
import torch
import torchvision
from torchvision .models._utils import IntermediateLayerGetter
model = torchvision.models.resnet50(pretrained=False) # 获取resnet50模型
# 迭代输出resnet50模型的网络结构名和参数
for name,para in model.named_parameters():
    print(name,para)
# return_layers是个字典，Key代表了导入的模型的网络结构的名字，value是每一层对应的新名字
# 不管字典中对导入的网络的结构顺序是如何存放，模型在执行的时候始终按照网络原有的顺序
return_layers={'layer1':'0','layer2':'1','layer3':'2','layer4':'3'}
body = IntermediateLayerGetter(model,return_layers)
body['layer1'] # 按照导入网络的网络结构名来得到原网络在该key下的网络结构
input = torch.randn(1,3,256,256)
# body(input) 按照导入的网络结构对输入一次操作，output里以字典return_layers里的V为名字，记录了每一次的输出，可以通过output['0']、output['1']提取经过相应网络结构的输出
output = body(input) 
output['0']
[print(k,v.shape) for k,v in output.items()]
# 经过网络的output包含了网络每一层结构的名字和每一层的输出
# output.items()进行访问
```

![](C:\Users\lenovo\Desktop\noteOf_languageAnd_tools\PyTorch_learnNotebook\Python_Debug_Image\gettttt.png)

##### 可选类型Optional[Tensor]

```python
Optional[Tensor] # 表明既可以传Tensor类型，也可以传None
m[None] # 不对数据做任何操作，但空间维度增加一维
```

