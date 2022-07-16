# Attention

预测结果时，把注意力放在不同的特征上。

对于word embedding，就是将单词word映射到另外一个空间，用n维的向量表示。

将每个单词用vector表示，vector的dimension表达着这个单词的属性，意思相近单词，dimension就会呈现出来。vector就是word embedding。

key把query作为依据，经过计算，得到对于query每个key的attention权重

*Q*,*K*,*V*三者再各应用中不一定相同，但是对于Transformer中的Self-Attention，三者取值都一样，就是取输入文本的embedding表示。

## 各种Attention

```python
# ExternalAttention
from attention.ExternalAttention import ExternalAttention
import torch
input=torch.randn(50,49,512)
ea = ExternalAttention(d_model=512,S=8)
output=ea(input)
print(output.shape)
```

```python
# Self-Attention 用于计算特征中不同位置之间的权重，从而达到更新特征的效果
from attention.SelfAttention import ScaledDotProductAttention
import torch
input = torch.randn(50,49,512)
sa = ScaledDotProductAttention(d_model=512,d_k=512,d_v=512,h=8)
output = sa(input,input,input)
print(output.shape)
```

```python
# Squeeze-and-Excitation(SE) Attention它通过对特征通道间的相关性进行建模，把重要的特征进行强化来提升准确率
from attention.SEAttention import SEAttention
import torch

input=torch.randn(50,512,7,7)
se = SEAttention(channel=512,reduction=8)
output=se(input)
print(output.shape)
```

```python
# Selective Kernel(SK)  Attention 通过动态计算每个卷积核得到通道的权重，动态的将各个卷积核的结果进行融合。
from attention.SKAttention import SKAttention
import torch

input=torch.randn(50,512,7,7)
se = SKAttention(channel=512,reduction=8)
output=se(input)
print(output.shape)
```

```python
# CBAM Attention
from attention.CBAM import CBAMBlock
import torch

input=torch.randn(50,512,7,7)
kernel_size=input.shape[2]
cbam = CBAMBlock(channel=512,reduction=16,kernel_size=kernel_size)
output=cbam(input)
print(output.shape)
```

```python
# BAM Attention
from attention.BAM import BAMBlock
import torch

input=torch.randn(50,512,7,7)
bam = BAMBlock(channel=512,reduction=16,dia_val=2)
output=bam(input)
print(output.shape)
```

```python
# ECA Attention
from attention.ECAAttention import ECAAttention
import torch

input=torch.randn(50,512,7,7)
eca = ECAAttention(kernel_size=3)
output=eca(input)
print(output.shape)
```

```python
# DANet Attention
#将self-attention用到场景分割的任务中，不同的是self-attention是关注每个position之间的注意力，而本文将self-attention做了一个拓展，还做了一个通道注意力的分支，操作上和self-attention一样，不同的通道attention中把生成Q，K，V的三个Linear去掉了。最后将两个attention之后的特征进行element-wise sum。
from attention.DANet import DAModule
import torch

input=torch.randn(50,512,7,7)
danet=DAModule(d_model=512,kernel_size=3,H=7,W=7)
print(danet(input).shape)
```

```python
# Pyramid Split Attention
from attention.PSA import PSA
import torch

input=torch.randn(50,512,7,7)
psa = PSA(channel=512,reduction=8)
output=psa(input)
print(output.shape)
```

```python
# Efficient Multi-Head Self-Attention
# 本文解决的主要是SA的两个痛点问题：（1）Self-Attention的计算复杂度和n（n为空间维度的大小）呈平方关系；（2）每个head只有q,k,v的部分信息，如果q,k,v的维度太小，那么就会导致获取不到连续的信息，从而导致性能损失
# 这篇文章给出的思路也非常简单，在SA中，在FC之前，用了一个卷积来降低了空间的维度，从而得到空间维度上更小的K和V。
from attention.EMSA import EMSA
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,64,512)
emsa = EMSA(d_model=512, d_k=512, d_v=512, h=8,H=8,W=8,ratio=2,apply_transform=True)
output=emsa(input,input,input)
print(output.shape)
```

