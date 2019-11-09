import argparse
import torch
import os
import cv2
import math
import sys
sys.path.append('.')
from darknet import DarkNet
import pickle as pkl
import random
from utils import *

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", dest='images', help="Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    parser.add_argument("--det", dest='det', help="Image / Directory to store detections to",default="det", type=str)
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1,type=int)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5,type=float)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4,type=float)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",default="cfg/yolov3.weights", type=str)
    parser.add_argument("--input_dim", dest='input_dim', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    return parser.parse_args()

args=arg_parse()
input_dim=int(args.input_dim)
assert input_dim%32==0 and input_dim>32 #输入dim最好是32的整数倍，因为yolo全程5次下采样
batchsize=args.bs
images_path=args.images
cfgFilepath=args.cfgfile
weightFilepath=args.weightsfile
CUDA=torch.cuda.is_available()

#加载类别信息
classes=load_classes("data/coco.names")
num_classes=len(classes)

#加载模型信息
model=DarkNet(cfgFilepath)
model.loadweight(weightFilepath)
model.netinfo['height']=input_dim
if CUDA:
    model.cuda()

#加载数据集
try:
    images=[os.path.join(os.path.realpath('.'),images_path,image) for image in os.listdir(images_path)]
except NotADirectoryError:
    images=[]
    images.append(os.path.join(os.path.realpath('.'),images_path))
except FileNotFoundError:
    print('{} file not exist!'.format(images_path))
    exit()

if not os.path.exists(args.det):
    os.makedirs(args.det)
#图像预处理
images_array=[cv2.imread(x) for x in images]
images_tensor=list(map(images_preprocess,images_array,[input_dim for i in range(len(images_array))]))
origin_images_dim=[(x.shape[1],x.shape[0]) for x in images_array] # w,h的顺序 与网络输出对应
origin_images_dim=torch.FloatTensor(origin_images_dim)

#划分batch
batchNum=math.ceil(len(images_tensor)/batchsize)
batch_images=[torch.cat(images_tensor[i*batchsize:min((i+1)*batchsize,len(images_tensor))],dim=0)
                     for i in range(batchNum)]

#测试
model.eval()
write=0
for batchidx,batchData in enumerate(batch_images):
    if CUDA:
        batchData=batchData.cuda()
    with torch.no_grad():
        prediction=model(batchData,CUDA)
    result=get_result(prediction,args.confidence,args.nms_thresh)

    if type(result)==int:
        continue
    result[:,0]+=batchidx*batchsize
    if write==0:
        output=result
        write=1
    else:
        output=torch.cat((output,result),dim=0)

    for i,image in enumerate(images[batchidx*batchsize:min((batchidx+1)*batchsize,len(images_tensor))]):
        index=i+batchidx*batchsize
        cls=[classes[int(out[-1])] for out in output if int(out[0])==index]
        print('picture:{}'.format(image.split(os.sep)[-1]))
        print('object detected:{}'.format(" ".join(cls)))
#检查是否存在检测结果
try:
    output
except NameError:
    print('output no exiting')
    exit()

#得到在原图片中的坐标位置 目前坐标位置是根据输入dim确定的
if CUDA:
    origin_images_dim=origin_images_dim.cuda()
origin_images_dim=torch.index_select(origin_images_dim,dim=0,index=output[:,0].long())

scale_factor=torch.min(input_dim/origin_images_dim,dim=1)[0].view(-1,1)
output[:,[1,3]]-=(input_dim-scale_factor*origin_images_dim[:,0].view(-1,1))/2
output[:,[2,4]]-=(input_dim-scale_factor*origin_images_dim[:,1].view(-1,1))/2

output[:,1:5]/=scale_factor
for i in range(output.shape[0]):
    output[:,[1,3]]=torch.clamp(output[:,[1,3]],min=0,max=origin_images_dim[i,0])
    output[:,[2,4]]=torch.clamp(output[:,[2,4]],min=0,max=origin_images_dim[i,1])


#绘制结果
colors = pkl.load(open("pallete", "rb"))
def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

list(map(lambda x:write(x,images_array) ,output))
images_name=list(map(lambda x:"{}/det_{}".format(args.det,x.split(os.sep)[-1]),images))
list(map(cv2.imwrite,images_name,images_array))
torch.cuda.empty_cache()
