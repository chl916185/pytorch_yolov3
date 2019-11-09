import torch
import numpy as np
import cv2

def load_classes(path):
    with open(path,'r') as f:
        names=f.read().split('\n')[:-1]
    return names

def predict_transfrom(prediction,anchors,num_classes,input_dim,CUDA):
    batch_size=prediction.shape[0]
    stride=input_dim//prediction.shape[2]
    grid_size=input_dim//stride
    num_property=num_classes+5
    num_anchors=len(anchors)

    prediction = prediction.view(batch_size, num_property * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction=prediction.view(batch_size,grid_size*grid_size*num_anchors,num_property)
    prediction[:,:,0]=torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1]=torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4]=torch.sigmoid(prediction[:,:,4])

    grid=np.arange(grid_size)
    x,y=np.meshgrid(grid,grid)
    x=torch.FloatTensor(x).view(-1,1)
    y=torch.FloatTensor(y).view(-1,1)
    if CUDA:
        x,y=x.cuda(),y.cuda()
    offset=torch.cat((x,y),dim=1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    prediction[:,:,:2]+=offset

    anchors=[(anchor[0]/stride,anchor[1]/stride) for anchor in anchors]
    anchors=torch.FloatTensor(anchors)
    if CUDA:
        anchors=anchors.cuda()
    anchors=anchors.repeat(grid_size*grid_size,1).unsqueeze(0)
    prediction[:,:,2:4]=torch.exp(prediction[:,:,2:4])*anchors

    prediction[:,:,5:]=torch.sigmoid(prediction[:,:,5:])
    prediction[:,:,:4]*=stride
    return prediction

def images_preprocess(images,input_dim):
    #保持原图纵横比不变
    h,w=images.shape[0],images.shape[1]
    new_h=int(h*min(input_dim/h,input_dim/w))
    new_w=int(w*min(input_dim/h,input_dim/w))
    temp=cv2.resize(images,(new_w,new_h),interpolation=cv2.INTER_CUBIC)
    images=np.full((input_dim,input_dim,3),128)
    images[(input_dim-new_h)//2:(input_dim-new_h)//2+new_h,(input_dim-new_w)//2:(input_dim-new_w)//2+new_w,:]=temp

    images=images[:,:,::-1].transpose((2,0,1)).copy()
    images_tensor=torch.from_numpy(images).float().div(255.0).unsqueeze(0)
    return images_tensor

def bbox_iou(box1,box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) \
                 * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,min=0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def get_result(prediction,confidence,nms_thresh=0.4):
    mask=(prediction[:,:,4]>confidence).float().unsqueeze(2)
    prediction*=mask

    #坐标转换
    box_corner=prediction.new(prediction.shape)
    box_corner[:,:,0]=prediction[:,:,0]-prediction[:,:,2]/2
    box_corner[:,:,1]=prediction[:,:,1]-prediction[:,:,3]/2
    box_corner[:,:,2]=prediction[:,:,0]+prediction[:,:,2]/2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:,:,:4]=box_corner[:,:,:4]

    write=0
    batchnum=prediction.shape[0]
    for index in range(batchnum):
        data=prediction[index]

        #数据格式转换
        max_score,max_class=torch.max(data[:,5:],dim=1)
        max_score=max_score.float().unsqueeze(1)
        max_class=max_class.float().unsqueeze(1)
        data=torch.cat((data[:,:5],max_score,max_class),dim=1)

        #消除不满足置信度的数据
        mask=torch.nonzero(data[:,4]).squeeze()
        try:
            data=data[mask,:].view(-1,7)
        except:
            continue
        if data.shape[0]==0:
            continue

        #对每个类别选择框 nms
        classes=torch.unique(data[:,-1])
        for cls in classes:
            mask=(data[:,-1]==cls).float().unsqueeze(1)
            temp=data*mask
            no_zero_id=torch.nonzero(temp[:,4]).squeeze()
            temp=temp[no_zero_id].view(-1,7)

            sorted_index=torch.sort(temp[:,4],descending=True)[1]
            temp=temp[sorted_index]

            num=temp.shape[0]
            for n in range(num):
                try:
                    iou=bbox_iou(temp[n].unsqueeze(0),temp[n+1:])
                except ValueError:
                    break
                except IndexError:
                    break

                mask=(iou<nms_thresh).float().unsqueeze(1)
                temp[n+1:,:]*=mask
                no_zero_id=torch.nonzero(temp[:,4]).squeeze()
                temp=temp[no_zero_id].view(-1,7)

            #将index加入到输出结果中
            batchid=temp.new(temp.shape[0],1).fill_(index)
            seq=(batchid,temp)
            if write==0:
                output=torch.cat(seq,dim=1)
                write=1
            else:
                out=torch.cat(seq,dim=1)
                output=torch.cat((output,out),dim=0)
    try:
        return output
    except:
        return 0
