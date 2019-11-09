import torch
from torch import nn
import numpy as np
from utils import *

#解析配置文件
def parse_cfg(filepath):
    with open(filepath,'r') as f:
        content=f.read().split('\n')
    content=[c for c in content if len(c)>0]
    content=[c.strip() for c in content if c[0]!='#']

    blocks=[]
    block={}
    for c in content:
        if c[0]=='[':
            if len(block)!=0:
                blocks.append(block)
                block={}
            block['type']=c[1:-1].strip()
        else:
            key,value=c.split('=')
            block[key.strip()]=value.strip()
    blocks.append(block)
    return blocks

#定义一个空层
class EmptyLayers(nn.Module):
    def __init__(self):
        super(EmptyLayers, self).__init__()

class DetectLayers(nn.Module):
    def __init__(self,anchors):
        super(DetectLayers, self).__init__()
        self.anchors=anchors

#生成网络模块
def createModule(blocks):
    netinfo=blocks[0]
    moduleList=nn.ModuleList()
    prev_filters=3
    output_filters=[] #记录每个输出层的filter

    for i,x in enumerate(blocks[1:]):
        module=nn.Sequential()
        module_type=x['type']
        #卷积层
        if module_type=='convolutional':
            activation=x['activation']
            try:
                batchnorm=int(x['batch_normalize'])
                bias=False
            except:
                batchnorm=0
                bias=True

            filters = int(x['filters'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            pad = int(x['pad'])

            if pad:
                padding=(kernel_size-1)//2
            else:
                padding=0
            conv=nn.Conv2d(in_channels=prev_filters,out_channels=filters,kernel_size=kernel_size,stride=stride,
                           padding=padding,bias=bias)
            module.add_module('conv2d_{}'.format(i),conv)

            if batchnorm:
                bn=nn.BatchNorm2d(filters)
                module.add_module('batchnorm_{}'.format(i),bn)

            if activation=='leaky':
                activt=nn.LeakyReLU(0.1,inplace=True)
                module.add_module('leakyrelu_{}'.format(i),activt)
        #定义短接层
        elif module_type=='shortcut':
            shortcut=EmptyLayers()
            module.add_module('shortcut_{}'.format(i),shortcut)
        #定义路由层
        elif module_type=='route':
            layers=x['layers'].split(',')
            start=int(layers[0])
            try:
                end=int(layers[1])
            except:
                end=0
            if start>0:
                start=start-i
            if end>0:
                end=end-i
            route = EmptyLayers()
            module.add_module('route_{}'.format(i), route)
            if end<0:
                filters=output_filters[i+start]+output_filters[i+end]
            else:
                filters=output_filters[i+start]

        #定义上采样层
        elif module_type=='upsample':
            upsample=nn.Upsample(scale_factor=2,mode='nearest')
            module.add_module('upsample_{}'.format(i),upsample)
        #定义yolo层
        elif module_type=='yolo':
            mask=x['mask'].split(',')
            mask=[int(i) for i in mask]

            anchors=list(map(int,x['anchors'].split(',')))
            anchors=list(zip(anchors[:-1:2],anchors[1::2]))

            anchors=[anchors[i] for i in mask]

            detection=DetectLayers(anchors)
            module.add_module('detectlayer_{}'.format(i),detection)

        moduleList.append(module)
        prev_filters=filters
        output_filters.append(filters)
    return (netinfo,moduleList)

class DarkNet(nn.Module):
    def __init__(self,cfgpath):
        super(DarkNet, self).__init__()
        self.blocks=parse_cfg(cfgpath)
        self.netinfo,self.modulelist=createModule(self.blocks)

    def forward(self, x,CUDA=True):
        modules=self.blocks[1:]
        output={} #记录每一层的输出，route、shortcut层需要使用
        write=0

        for i,module in enumerate(modules):
            module_type=module['type']
            if module_type=='convolutional' or module_type=='upsample':
                x=self.modulelist[i](x)
            elif module_type=='route':
                layers=module['layers'].split(',')
                layers=list(map(int,layers))

                start=layers[0]
                if start>0:
                    start=start-i
                if len(layers)==1:
                    x=output[start+i]
                else:
                    end=layers[1]
                    if end>0:
                        end=end-i
                    x1=output[start+i]
                    x2=output[end+i]
                    x=torch.cat((x1,x2),dim=1)
            elif module_type=='shortcut':
                from_=int(module['from'])
                x=output[i-1]+output[i+from_]
            elif module_type=='yolo':
                anchors=self.modulelist[i][0].anchors
                input_dim=int(self.netinfo['height'])
                num_classes=int(module['classes'])

                #特征图转换
                x=x.data
                x=predict_transfrom(x,anchors,num_classes,input_dim,CUDA)
                if write==0:
                    detection=x
                    write=1
                else:
                    detection=torch.cat((detection,x),dim=1)
            output[i]=x
        return detection

    def loadweight(self,weightfile):
        fp=open(weightfile,'rb')
        # The first 5 values are header information
        header=np.fromfile(fp,dtype=np.int32,count=5)
        weights=np.fromfile(fp,dtype=np.float32)

        ptr=0
        for i,module in enumerate(self.modulelist):
            module_type=self.blocks[i+1]['type']
            if module_type=='convolutional':
                try:
                    batch_normalize=int(self.blocks[i+1]['batch_normalize'])
                except:
                    batch_normalize=0

                conv=module[0]
                if batch_normalize:
                    bn=module[1]
                    num_bn_biases = bn.bias.numel()

                    bn_biases=torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr+=num_bn_biases
                    bn_weights=torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr+=num_bn_biases
                    bn_running_mean=torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr+=num_bn_biases
                    bn_running_var=torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                    ptr+=num_bn_biases

                    bn_biases=bn_biases.view_as(bn.bias.data)
                    bn_weights=bn_weights.view_as(bn.weight.data)
                    bn_running_mean=bn_running_mean.view_as(bn.running_mean)
                    bn_running_var=bn_running_var.view_as(bn.running_var)

                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    num_conv_biases=conv.bias.numel()
                    conv_biases=torch.from_numpy(weights[ptr:ptr+num_conv_biases])
                    ptr+=num_conv_biases

                    conv_biases=conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)

                num_conv_weights=conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_conv_weights])
                ptr += num_conv_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


